#!/usr/bin/env python3
"""Run the fine-tuned model (and optionally the base model) on images you give, and get the JSON back.

Each image is one document. The fields to extract come from, in this order:
  --gt-dir   a folder with the ground truth <image name>.json  (also scores the answers)
  --schema   a JSON file whose keys are the fields                (same fields for every image)
  --fields   a comma-separated list of field names                (same fields for every image)
The prompt, image size and decoding are exactly those of infer.py / training (settings read from the adapter).

Examples
  # one image, fields from its ground truth, fine-tuned and base side by side
  python scripts/infer_bench.py dataset/forms/benchmark/images/form_012.png \
      --gt-dir dataset/forms/benchmark/gt_json --compare-base --out runs/demo

  # three images, also show saved Gemini answers (<image name>.json) for the same documents
  python scripts/infer_bench.py img1.png img2.png img3.png --gt-dir gt/ --gemini-dir gemini/ --compare-base --out runs/demo

  # no ground truth: give the field names
  python scripts/infer_bench.py cheque.jpg --fields "Payee,Amount in words,Amount in figures,Date" --out runs/demo

Writes to --out: <image>_<model>.json (the parsed JSON, or the raw text if it is not valid JSON),
results.json (everything, with scores when a ground truth exists) and report.html (image + field table).
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
import infer as I  # noqa: E402  (also imports train.py: same prompt, processor and model loading)


def fields_for(image: Path, a) -> tuple[dict[str, Any], Optional[Any]]:
    """(schema, ground truth or None) for one image, built the way training builds it."""
    target = None
    if a.gt_dir:
        gt_file = Path(a.gt_dir) / f"{image.stem}.json"
        if not gt_file.is_file():
            raise SystemExit(f"No ground truth {gt_file} for {image.name} (names must match).")
        obj = json.loads(gt_file.read_text(encoding="utf-8"))
        target = obj["target"] if isinstance(obj, dict) and isinstance(obj.get("target"), dict) else obj
        schema = {k: "verbatim-string" for k in target}                # as train.load_records does
    elif a.schema:
        schema = json.loads(Path(a.schema).read_text(encoding="utf-8"))
    elif a.fields:
        schema = {f.strip(): "verbatim-string" for f in a.fields.split(",") if f.strip()}
    else:
        raise SystemExit("Give the fields: --gt-dir, --schema or --fields.")
    return I.normalize_template(schema), target


def field_rows(target: Any, parsed: Any, mc: dict[str, Any]) -> dict[str, tuple[Optional[str], bool]]:
    """GT field -> (predicted value, correct?) with the metric's own key matching and normalisation."""
    gt = I.flatten(target)
    pred = I.flatten(parsed) if isinstance(parsed, (dict, list)) else {}
    matches, missing, _ = I.align_keys(gt, pred, mc["alias_lookup"], mc["mode"], mc["threshold"])
    rows = {g: (pred[p], I.normalize_value(gt[g]) == I.normalize_value(pred[p])) for g, p in matches}
    rows.update({g: (None, False) for g in missing})
    return rows


def f1(score: dict[str, Any]) -> float:
    tp, fp, fn = score["tp"], score["fp"], score["fn"]
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 100 * 2 * p * r / (p + r) if p + r else 0.0


def thumb(path: Path, width: int = 700) -> str:
    try:
        from PIL import Image
        with Image.open(path) as im:
            im = im.convert("RGB")
            if im.width > width:
                im = im.resize((width, int(im.height * width / im.width)))
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return ""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+", type=Path, help="image files; each one is a separate document")
    ap.add_argument("--adapter", default="runs/qwen25vl_B4/final_adapter")
    ap.add_argument("--gt-dir", default=None, help="folder with <image name>.json ground truths (fields + scoring)")
    ap.add_argument("--schema", default=None, help="JSON file whose keys are the fields to extract")
    ap.add_argument("--fields", default=None, help="comma-separated field names")
    ap.add_argument("--gemini-dir", default=None, help="folder with saved Gemini answers <image name>.json (optional)")
    ap.add_argument("--compare-base", action="store_true", help="also run the base model (adapter switched off)")
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--metric-mode", default="schema", choices=["schema", "schemafree", "auto"])
    ap.add_argument("--out", type=Path, default=Path("runs/demo"))
    a = ap.parse_args()

    for img in a.images:
        if not img.is_file():
            raise SystemExit(f"Image not found: {img}")
    a.out.mkdir(parents=True, exist_ok=True)

    # same settings as training, read from the adapter (model, image budget, revision)
    st = I.resolve_settings(SimpleNamespace(adapter=a.adapter, model=None, max_pixels=None, min_pixels=None,
                                            revision=None))
    mc = I.make_metric_config(SimpleNamespace(metric_mode=a.metric_mode, key_threshold=0.80, aliases=None))
    docs = []
    for img in a.images:
        schema, target = fields_for(img, a)
        docs.append({"id": img.stem, "images": [str(img.resolve())], "schema": schema,
                     "schema_text": I.schema_to_text(schema), "target": target})

    print(f"Loading {I.T.SPEC.alias} + adapter {a.adapter} (image budget {st['max_pixels']} tokens per page) ...",
          flush=True)
    from peft import PeftModel
    processor = I.load_processor(st["max_pixels"], st["min_pixels"], st["revision"])
    model = PeftModel.from_pretrained(I.load_base_model(st["revision"]), a.adapter, is_trainable=False)
    model.eval()
    model.config.use_cache = True
    tc = getattr(model.config, "text_config", None)
    if tc is not None and hasattr(tc, "use_cache"):
        tc.use_cache = True

    variants = ["finetuned"] + (["base"] if a.compare_base else [])
    results = []
    for d in docs:
        res: dict[str, Any] = {"image": d["images"][0], "fields": list(d["schema"]), "target": d["target"]}
        for v in variants:
            t0 = time.time()
            ctx = model.disable_adapter() if v == "base" else I.nullcontext()
            with ctx:
                raw = I.generate_with_backoff(model, processor, [d], a.max_new_tokens)[0]
            res[v] = {"raw": raw, "parsed": I.parse_json_answer(raw), "seconds": round(time.time() - t0, 1)}
        if a.gemini_dir:
            gfile = Path(a.gemini_dir) / f"{d['id']}.json"
            raw = gfile.read_text(encoding="utf-8-sig") if gfile.is_file() else ""
            res["gemini"] = {"raw": raw, "parsed": I.parse_json_answer(raw), "missing_file": not gfile.is_file()}
        models = variants + (["gemini"] if a.gemini_dir else [])
        for v in models:
            out_obj = res[v]["parsed"] if res[v]["parsed"] is not None else {"_invalid_json_raw_output": res[v]["raw"]}
            (a.out / f"{d['id']}_{v}.json").write_text(json.dumps(out_obj, indent=2, ensure_ascii=False),
                                                     encoding="utf-8")
            if d["target"] is not None:
                res[v]["score"] = I.score_json(res[v]["parsed"], d["target"], mc)
                res[v]["field_rows"] = field_rows(d["target"], res[v]["parsed"], mc)

        # ---- console
        print(f"\n=== {Path(d['images'][0]).name} ===")
        for v in models:
            ok = ("no saved answer file" if res[v].get("missing_file")
                  else "valid JSON" if res[v]["parsed"] is not None else "NOT valid JSON")
            extra = f", {res[v]['seconds']} s" if "seconds" in res[v] else ""
            sc = res[v].get("score")
            scs = f" | F1 {f1(sc):.1f}%, strict {'yes' if sc['strict_form_correct'] else 'no'}" if sc else ""
            print(f"\n[{v}] {ok}{extra}{scs}")
            print(json.dumps(res[v]["parsed"], indent=2, ensure_ascii=False) if res[v]["parsed"] is not None
                  else res[v]["raw"][-800:])
        if d["target"] is not None:
            print("\nfield-by-field (OK = correct after the metric's normalisation):")
            print(f"  {'field':30s} {'ground truth':28s} " + " ".join(f"{v:28s}" for v in models))
            for k, gv in I.flatten(d["target"]).items():
                cells = []
                for v in models:
                    val, good = res[v]["field_rows"].get(k, (None, False))
                    cells.append(f"{('OK ' if good else 'XX ') + ('(missing)' if val is None else str(val))[:25]:28s}")
                print(f"  {k[:30]:30s} {str(gv)[:28]:28s} " + " ".join(cells))
        results.append(res)

    # ---- files
    (a.out / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str),
                                        encoding="utf-8")
    css = ("body{font-family:system-ui,sans-serif;margin:24px;color:#1b1f24;background:#fff}table{border-collapse:collapse;"
           "font-size:13px}td,th{border:1px solid #d0d7de;padding:4px 8px;vertical-align:top;max-width:260px;"
           "word-break:break-word}th{background:#f3f4f6}.ok{background:#dcfce7}.bad{background:#fee2e2}"
           ".wrap{display:flex;gap:24px;flex-wrap:wrap;align-items:flex-start}img{border:1px solid #d0d7de;max-width:100%}"
           "pre{background:#f6f8fa;padding:8px;font-size:12px;white-space:pre-wrap;max-width:520px}.meta{color:#4b5563}")
    parts = [f"<!doctype html><meta charset='utf-8'><title>Extraction results</title><style>{css}</style>",
             f"<h1>Extraction results</h1><p class='meta'>Adapter {html.escape(a.adapter)}; image budget "
             f"{st['max_pixels']} tokens per page; greedy decoding; key matching: {a.metric_mode}.</p>"]
    for res in results:
        name = Path(res["image"]).name
        models = [m for m in ("base", "finetuned", "gemini") if m in res]
        head = " | ".join(f"{m}: " + ("no saved answer" if res[m].get("missing_file") else
                                       "valid JSON" if res[m]["parsed"] is not None else "INVALID JSON")
                          + (f", F1 {f1(res[m]['score']):.1f}%" if "score" in res[m] else "") for m in models)
        parts.append(f"<h2>{html.escape(name)}</h2><p class='meta'>{html.escape(head)}</p><div class='wrap'>")
        b64 = thumb(Path(res["image"]))
        if b64:
            parts.append(f"<img src='data:image/jpeg;base64,{b64}' width='520'>")
        if res["target"] is not None:
            parts.append("<table><tr><th>field</th><th>ground truth</th>"
                         + "".join(f"<th>{m}</th>" for m in models) + "</tr>")
            for k, gv in I.flatten(res["target"]).items():
                parts.append(f"<tr><td>{html.escape(k)}</td><td>{html.escape(str(gv))}</td>")
                for m in models:
                    val, good = res[m]["field_rows"].get(k, (None, False))
                    parts.append(f"<td class='{'ok' if good else 'bad'}'>"
                                 f"{'(missing)' if val is None else html.escape(str(val))}</td>")
                parts.append("</tr>")
            parts.append("</table>")
        else:
            for m in models:
                body = res[m]["parsed"] if res[m]["parsed"] is not None else res[m]["raw"]
                parts.append(f"<div><b>{m}</b><pre>{html.escape(json.dumps(body, indent=2, ensure_ascii=False) if not isinstance(body, str) else body)}</pre></div>")
        parts.append("</div>")
    (a.out / "report.html").write_text("\n".join(parts), encoding="utf-8")
    print(f"\nSaved: {a.out}/<image>_<model>.json, {a.out}/results.json, {a.out}/report.html")


if __name__ == "__main__":
    main()
