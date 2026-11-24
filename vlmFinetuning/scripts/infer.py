#!/usr/bin/env python3
"""Inference and scoring for LoRA adapters trained with train.py (all three models), or for a base model.

Prompt rendering, image loading and batching are imported from train.py, so the model sees exactly the token
layout it was trained on. Keep both files in the same folder.

Input modes
  1. One document:     --images page1.png --schema schema.json
  2. A saved split:    --data prepared/manifest.jsonl --split validation|test   (IDs from the run's splits.json)
  3. Any manifest:     --data prepared/legacy_forms.jsonl --split all            ('target' optional)

Metrics (same definitions as Metric_new.py, code included below; in %): Combined Key-Value Precision, Recall, F1;
whole-form JSON similarity (FormSim); Strict Form Accuracy.
  FormSim is a CUSTOM whole-form measure, not a plain text edit distance: both forms are flattened, keys and values
  normalised, the (key, value) pairs sorted and serialised as JSON, and the similarity is
  1 - Levenshtein(serialised GT, serialised prediction) / max length. Keys and JSON structure count, not only the
  extracted text. (Stored as normalized_edit_distance_similarity_pct for compatibility.) Key matching defaults to --metric-mode schema (exact key names after
normalisation, plus the alias table; no fuzzy matching). Use the SAME --metric-mode and --aliases for every model
you compare (fine-tuned, base, external models scored with Metric_new.py).

External predictions (e.g. Gemini): --external-predictions <folder of JSON files | JSONL> --external-name gemini
  scores another model's saved outputs on the SAME documents (same split IDs, same manifest checks) with the SAME
  scoring rules and aliases, without loading any model. Missing predictions count as empty (all fields wrong) and are
  listed in the summary.

Best checkpoint: with --adapter, the run's best checkpoint (training_config.json) is compared tensor by tensor with
final_adapter/ when that checkpoint folder still exists; a mismatch stops the run.

Output: an output folder that already holds results for this split is not overwritten (use a new --out, or
--overwrite).

Safety checks for scored runs (all stop the run):
  - --split validation|test: the manifest must be the one the split was made from (manifest_sha256 in splits.json),
    and every image must have the same bytes as at training time (<run>/image_hashes.json, when present);
  - every record must have a ground truth ('target') and a 'dataset' label, and every record must be scored
    (--allow-unlabeled turns this off for prediction-only runs). Results are reported overall AND per dataset, with the number of documents; a dataset with fewer than
--min-docs documents is marked PRELIMINARY. The model and its settings are read from <adapter>/training_config.json;
for a base model pass --model.

Examples
  python scripts/infer.py --adapter runs/qwen25vl_C/final_adapter --data prepared/manifest.jsonl --split validation
  python scripts/infer.py --adapter runs/qwen25vl_C/final_adapter --data prepared/manifest.jsonl --split test
  python scripts/infer.py --model glm-4.6v-flash --data prepared/legacy_cheques.jsonl --split all --out runs/base_glm46v/legacy_cheques
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter
from datetime import date
from decimal import Decimal, InvalidOperation
from difflib import SequenceMatcher
from types import SimpleNamespace
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Optional

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train as T  # noqa: E402
from train import (  # noqa: E402
    LOG, MODEL_REGISTRY, collate_encoded, encode_prompt, enforce_lock, get_tokenizer, load_base_model, load_processor,
    load_records, normalize_template, package_versions, resolve_model_revision, schema_to_text, setup_run_logging,
    use_model,
)


# =============================================================================
# 1. SETTINGS AND INPUT RECORDS
# =============================================================================
def resolve_settings(args) -> dict[str, Any]:
    cfg: dict[str, Any] = {}
    if args.adapter:
        cfg_path = Path(args.adapter) / "training_config.json"
        if not cfg_path.is_file():
            raise SystemExit(f"{cfg_path} not found: not an adapter made by train.py.")
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        if args.model and args.model != cfg.get("model_alias"):
            raise SystemExit(f"This adapter was trained for {cfg.get('model_alias')}; you passed --model {args.model}.")
        args.model = cfg.get("model_alias")
    if not args.model:
        raise SystemExit(f"Pass --adapter, or --model for a base model ({sorted(MODEL_REGISTRY)}).")
    SPEC = use_model(args.model)
    if args.adapter and cfg.get("base_model") != SPEC.hf_id:
        raise SystemExit(f"Adapter base model {cfg.get('base_model')} differs from {SPEC.hf_id}.")
    if args.adapter:                                 # (this check was unreachable before: it sat after the raise)
        trained = cfg.get("packages", {})
        now = package_versions()
        for name in ("transformers", "peft", "torch"):
            if trained.get(name) and trained.get(name) != now.get(name):
                LOG.warning("%s %s here, but the adapter was trained with %s.", name, now.get(name), trained[name])
    settings = {
        "max_pixels": args.max_pixels or cfg.get("max_pixels", T.SPEC.default_max_pixels),
        "min_pixels": args.min_pixels or cfg.get("min_pixels", 256),
        "revision": args.revision or cfg.get("revision"),
    }
    for key in ("max_pixels", "min_pixels", "revision"):
        if key in cfg and cfg[key] is not None and cfg[key] != settings[key]:
            LOG.warning("%s=%s differs from training (%s); results may not match validation.",
                        key, settings[key], cfg[key])
    return settings


def build_records(args) -> list[dict[str, Any]]:
    if args.images:
        if not args.schema:
            raise SystemExit("--images needs --schema (the field schema / template JSON).")
        schema = normalize_template(json.loads(Path(args.schema).read_text(encoding="utf-8")))
        paths = []
        for p in args.images:
            path = Path(p).resolve()
            if not path.is_file():
                raise SystemExit(f"Image not found: {path}")
            paths.append(str(path))
        return [{"id": args.doc_id, "images": paths, "schema": schema,
                 "schema_text": schema_to_text(schema), "target": None}]
    if not args.data:
        raise SystemExit("Pass either --images or --data.")
    records = load_records(args.data, args.schema, require_target=False)
    if args.split != "all":
        run_dir = Path(args.run_dir) if args.run_dir else (Path(args.adapter).resolve().parent if args.adapter else None)
        split_file = run_dir / "splits.json" if run_dir else None
        if split_file is None or not split_file.is_file():
            raise SystemExit(f"--split {args.split} needs splits.json from training (looked in {run_dir}); "
                             "set --run-dir or use --split all.")
        split_data = json.loads(split_file.read_text(encoding="utf-8"))
        ids = split_data.get(args.split, [])
        if not ids:
            raise SystemExit(f"Split {args.split!r} is empty in {split_file}.")
        verify_manifest(args.data, split_data.get("_meta", {}), split_file)
        by_id = {r["id"]: r for r in records}
        missing = [i for i in ids if i not in by_id]
        if missing:
            raise SystemExit(f"{len(missing)} split IDs are not in {args.data}, e.g. {missing[:3]}")
        records = [by_id[i] for i in ids]
        verify_image_bytes(records, run_dir / "image_hashes.json")
    records = records[: args.limit] if args.limit else records
    if not args.allow_unlabeled:
        unlabeled = [r["id"] for r in records if r.get("target") is None]
        no_dataset = [r["id"] for r in records if not r.get("dataset")]
        if unlabeled or no_dataset:
            raise SystemExit(f"{len(unlabeled)} record(s) without ground truth (e.g. {unlabeled[:3]}) and "
                             f"{len(no_dataset)} without a dataset label (e.g. {no_dataset[:3]}): a scored run must "
                             "cover every document. Fix the manifest, or pass --allow-unlabeled for predictions only.")
    return records


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_manifest(data_path: str, meta: dict[str, Any], split_file: Path) -> None:
    """The split lists IDs; make sure they point at the same manifest (same images and ground truths)."""
    expected = meta.get("manifest_sha256")
    if not expected:
        raise SystemExit(f"{split_file} has no manifest fingerprint (_meta.manifest_sha256): cannot verify that "
                         f"{data_path} is the manifest the split was made from.")
    have = file_sha256(Path(data_path))
    if have != expected:
        raise SystemExit(f"{data_path} (sha256 {have[:12]}) is NOT the manifest {split_file} was made from "
                         f"({expected[:12]}): images or ground truths may have changed since training.")
    LOG.info("Manifest fingerprint matches the split: %s", have[:12])


def verify_image_bytes(records: list[dict[str, Any]], hashes_file: Path) -> None:
    """Compare every image's bytes with the hashes train.py recorded (runs trained before 2026-11-23 have none)."""
    if not hashes_file.is_file():
        LOG.warning("%s not found (run trained with an older train.py): image bytes not verified.", hashes_file)
        return
    known = json.loads(hashes_file.read_text(encoding="utf-8")).get("images", {})
    bad = []
    for r in records:
        for img in r["images"]:
            if known.get(img) != file_sha256(Path(img)):
                bad.append(img)
    if bad:
        raise SystemExit(f"{len(bad)} image(s) differ from training time or were not part of the run, "
                         f"e.g. {bad[:3]} (checked against {hashes_file}).")
    LOG.info("Image bytes match training time for all %d documents.", len(records))


def verify_best_adapter(adapter: str) -> dict[str, Any]:
    """final_adapter must be the best checkpoint (lowest validation loss): compare their LoRA tensors."""
    cfg = json.loads((Path(adapter) / "training_config.json").read_text(encoding="utf-8"))
    best = cfg.get("best_checkpoint")
    report: dict[str, Any] = {"best_checkpoint": best, "best_eval_loss": cfg.get("best_eval_loss"),
                              "global_step": cfg.get("global_step")}
    if not best:
        LOG.warning("training_config.json records no best checkpoint; cannot confirm final_adapter is the best one.")
        report["verified"] = None
        return report
    ckpt = Path(best)
    if not ckpt.is_absolute() and not ckpt.exists():
        ckpt = Path(adapter).resolve().parents[2] / best           # relative to the project folder (runs/<run>/...)
    a_file, c_file = Path(adapter) / "adapter_model.safetensors", ckpt / "adapter_model.safetensors"
    if not c_file.is_file():
        LOG.warning("Best checkpoint %s is no longer on disk; final_adapter not compared with it.", best)
        report["verified"] = None
        return report
    from safetensors.torch import load_file
    fa, fc = load_file(str(a_file)), load_file(str(c_file))
    same = fa.keys() == fc.keys() and all(torch.equal(fa[k], fc[k]) for k in fa)
    report["verified"] = same
    if not same:
        raise SystemExit(f"final_adapter differs from the best checkpoint {best}: use the adapter saved from the best "
                         "checkpoint.")
    LOG.info("final_adapter = best checkpoint %s (eval_loss %s, %d tensors identical).", best,
             cfg.get("best_eval_loss"), len(fa))
    return report


def load_external_predictions(path: str, records: list[dict[str, Any]]) -> tuple[list[Optional[str]], list[str]]:
    """Raw prediction text per record, from a folder of <image-name>.json files (as Metric_new.py reads them) or a
    JSONL with {"id": <record id>, "prediction"|"parsed"|"output"|"raw": ...}. Returns (texts, missing ids)."""
    src = Path(path)
    by_key: dict[str, list[Any]] = {}
    if src.is_dir():
        for f in sorted(src.rglob("*")):
            if f.is_file() and f.suffix.lower() in {".json", ".txt"}:
                by_key.setdefault(f.stem, []).append(f)
    elif src.is_file():
        for line in src.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            value = next((row[k] for k in ("prediction", "parsed", "output", "raw") if k in row), None)
            text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
            rid = str(row.get("id", ""))
            by_key.setdefault(rid, []).append(text)
            if Path(rid).name != rid:
                by_key.setdefault(Path(rid).name, []).append(text)
    else:
        raise SystemExit(f"--external-predictions {src} not found")
    texts, missing = [], []
    for r in records:
        keys = [r["id"], Path(r["id"]).name, Path(r["images"][0]).stem]
        hit = next((by_key[k] for k in keys if k in by_key), None)
        if hit is None:
            texts.append(None)
            missing.append(r["id"])
            continue
        if len(hit) > 1:
            raise SystemExit(f"{len(hit)} external predictions match {r['id']} ({hit[:2]}); make the names unique.")
        texts.append(hit[0].read_text(encoding="utf-8-sig") if isinstance(hit[0], Path) else hit[0])
    if missing:
        LOG.warning("%d of %d documents have no external prediction (scored as empty), e.g. %s",
                    len(missing), len(records), missing[:3])
    return texts, missing


# =============================================================================
# 2. GENERATION (batched, left-padded, with OOM back-off)
# =============================================================================
@torch.no_grad()
def generate_texts(model, processor, records: list[dict[str, Any]], max_new_tokens: int) -> list[str]:
    tok = get_tokenizer(processor)
    items = [encode_prompt(processor, r) for r in records]
    batch = collate_encoded(items, tok.pad_token_id, padding_side="left")
    device = next(model.parameters()).device
    batch = {k: v.to(device) for k, v in batch.items()}
    output = model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
                            pad_token_id=tok.pad_token_id)
    new_tokens = output[:, batch["input_ids"].shape[1]:]
    for r, row in zip(records, new_tokens):
        n_new = int((row != tok.pad_token_id).sum())
        if n_new >= max_new_tokens:
            LOG.warning("%s: answer hit --max-new-tokens=%d and is probably cut off (invalid JSON); "
                        "raise --max-new-tokens.", r["id"], max_new_tokens)
    return [tok.decode(row, skip_special_tokens=True).strip() for row in new_tokens]


def generate_with_backoff(model, processor, records, max_new_tokens) -> list[str]:
    try:
        return generate_texts(model, processor, records, max_new_tokens)
    except torch.cuda.OutOfMemoryError:
        if len(records) == 1:
            raise
        torch.cuda.empty_cache()
        half = len(records) // 2
        LOG.warning("OOM with %d documents; retrying as %d + %d.", len(records), half, len(records) - half)
        return (generate_with_backoff(model, processor, records[:half], max_new_tokens)
                + generate_with_backoff(model, processor, records[half:], max_new_tokens))


# =============================================================================
# 3. PARSING AND SCORING (metrics from Metric_new.py, schema key matching by default)
# =============================================================================
def parse_json_answer(raw: str) -> Optional[Any]:
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.S)
    text = re.sub(r"^.*?</think>", "", text, flags=re.S)       # a closing tag without its opening one
    match = re.search(r"<answer>(.*?)</answer>", text, flags=re.S)
    if match:
        text = match.group(1)
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I | re.M)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


# Scoring = the metrics of Metric_new.py (its code is copied below): keys are aligned (exact, aliases, then lexical
# similarity in "auto" mode) and values compared after domain-aware normalisation (dates, numbers, currency, sex).
#   Combined KV Precision / Recall / F1  TP = aligned key with equal normalised value; a wrong value = 1 FP + 1 FN;
#                                        extra predicted key = FP; missing ground-truth key = FN. Micro-averaged.
#   FormSim (whole-form JSON similarity) 1 - Levenshtein(canonical GT form, canonical prediction form) / max length,
#                                        in %, macro-averaged over documents.
#   Strict Form Accuracy                 every GT key matched, no extra keys, every value identical after NFKC,
#                                        lowercasing and whitespace collapsing only. % of documents.
# Invalid JSON is scored as an empty prediction (every GT key missing).
# ---- copied from Metric_new.py (unchanged except the strict-key rule in evaluate_one) ------------------------------------------------------------
DEFAULT_ALIASES = {
    "patient_name": ["name_of_patient", "patients_name", "pt_name", "patient_full_name"],
    "age": ["patient_age", "age_years", "age_of_patient", "years_old"],
    "sex": ["gender", "patient_sex", "patient_gender"],
    "date_of_admission": ["admission_date", "admit_date", "date_admitted"],
    "date_of_discharge": ["discharge_date", "date_discharged"],
    "hospital_name": ["hospital", "name_of_hospital", "facility_name", "medical_center"],
    "account_number": ["account_no", "acc_no", "a_c_no", "ac_no", "account_num"],
    "cheque_number": ["cheque_no", "check_number", "check_no", "chq_no"],
    "amount_in_words": ["amount_words", "legal_amount"],
    "amount_in_figures": ["amount_figures", "courtesy_amount", "amount_numeric"],
    "payee_name": ["payee", "pay_to", "beneficiary_name"],
    "invoice_number": ["invoice_no", "inv_no", "bill_number", "bill_no"],
    "invoice_date": ["date_of_invoice", "bill_date"],
    "company": ["company_name", "vendor", "vendor_name", "seller", "store_name"],
    "address": ["company_address", "vendor_address", "store_address"],
    "total": ["total_amount", "grand_total", "amount_due", "net_total"],
    "document_type": ["doc_type", "document_class", "form_type", "document_category"],
}
WRAPPERS = {"fields", "data", "extracted_fields", "result", "results", "output",
            "key_value_pairs", "form", "extraction"}
PAIR_FIELDS = [("key", "value"), ("question", "answer"), ("field", "value"), ("label", "text")]
CONFLICT_PAIRS = [("admission", "discharge"), ("start", "end"), ("from", "to"),
                  ("issue", "due"), ("billing", "shipping"), ("sender", "receiver"),
                  ("payer", "payee"), ("debit", "credit"), ("first", "last"),
                  ("birth", "death"), ("subtotal", "total"), ("words", "figures"),
                  ("min", "max")]


def norm_key(x):
    s = unicodedata.normalize("NFKC", str(x))
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    return re.sub(r"[^0-9A-Za-z]+", "_", s).strip("_").casefold()


def leaf_key(k):
    return str(k).rsplit(".", 1)[-1].replace("]", "").split("[")[0]


def make_alias_lookup(aliases):
    result = {}
    for canonical, variants in aliases.items():
        c = norm_key(canonical)
        result[c] = c
        for alias in variants:
            result[norm_key(alias)] = c
    return result


def canonical_path(key, lookup):
    parts = re.split(r"([.\[\]])", str(key))
    return "".join(
        part if part in {".", "[", "]", ""} else lookup.get(norm_key(part), norm_key(part))
        for part in parts
    )


def scalar(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def canonical_number(value):
    try:
        number = Decimal(str(value))
        if not number.is_finite():
            return str(value)
        result = format(number.normalize(), "f")
        if "." in result:
            result = result.rstrip("0").rstrip(".")
        return "0" if result in {"", "-0"} else result
    except (InvalidOperation, ValueError):
        return str(value)


def normalize_value(value):
    s = unicodedata.normalize("NFKC", scalar(value)).strip().casefold()
    s = re.sub(r"\s+", " ", s)
    if s in {"", "n/a", "na", "none", "null", "nil", "-", "--"}:
        return ""
    if s in {"m", "male"}:
        return "male"
    if s in {"f", "female"}:
        return "female"
    age = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(?:years?|yrs?|y\.?o\.?)", s)
    if age:
        return canonical_number(age.group(1))
    dm = re.fullmatch(r"(\d{1,2})[/-](\d{1,2})[/-](\d{2}|\d{4})", s)
    if dm:
        day, month, year = map(int, dm.groups())
        if year < 100:
            year += 2000 if year <= 68 else 1900
        try:
            return date(year, month, day).isoformat()
        except ValueError:
            pass
    numeric = re.sub(r"^(?:rm|rs\.?|inr|usd|₹|\$)\s*", "", s).strip()
    compact = numeric.replace(",", "").replace(" ", "")
    if re.fullmatch(r"[+-]?\d+(?:\.\d+)?", compact):
        return canonical_number(compact)
    return re.sub(r"\s+", "", s)


def strict_text(value):
    """Only NFKC + lowercase + whitespace collapsing for Strict Form Accuracy."""
    s = unicodedata.normalize("NFKC", scalar(value)).casefold()
    return re.sub(r"\s+", " ", s).strip()


def unwrap(data):
    while isinstance(data, dict) and len(data) == 1:
        key, value = next(iter(data.items()))
        if norm_key(key) in WRAPPERS and isinstance(value, (dict, list)):
            data = value
        else:
            break
    return data


def flatten(obj, prefix=""):
    out = {}
    if isinstance(obj, dict):
        obj = unwrap(obj)
        for key, value in obj.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            out.update(flatten(value, child))
        if not obj and prefix:
            out[prefix] = ""
    elif isinstance(obj, list):
        if obj and all(not isinstance(x, (dict, list)) for x in obj):
            out[prefix or "__root__"] = " ".join(scalar(x) for x in obj)
        elif obj and all(isinstance(x, dict) for x in obj):
            for kf, vf in PAIR_FIELDS:
                if all(kf in item and vf in item for item in obj):
                    for i, item in enumerate(obj):
                        base = str(item[kf]).strip() or f"item_{i+1}"
                        name, n = base, 2
                        while name in out:
                            name, n = f"{base}_{n}", n + 1
                        out[name] = scalar(item[vf])
                    return out
            for i, item in enumerate(obj):
                out.update(flatten(item, f"{prefix}[{i}]" if prefix else f"item_{i}"))
        elif not obj and prefix:
            out[prefix] = ""
        else:
            for i, item in enumerate(obj):
                out.update(flatten(item, f"{prefix}[{i}]" if prefix else f"item_{i}"))
    else:
        out[prefix or "__root__"] = scalar(obj)
    return out


def load_flat(path):
    raw = path.read_text(encoding="utf-8-sig").strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.I)
    return flatten(json.loads(raw))


def has_conflicting_semantics(a, b):
    ta, tb = set(norm_key(a).split("_")), set(norm_key(b).split("_"))
    return any((x in ta and y in tb) or (y in ta and x in tb) for x, y in CONFLICT_PAIRS)


def align_keys(gt, pred, alias_lookup, mode, threshold):
    gt_left, pred_left, matches = list(gt), list(pred), []
    pred_groups = {}
    for p in pred_left:
        pred_groups.setdefault(canonical_path(p, alias_lookup), []).append(p)
    for g in list(gt_left):
        candidates = pred_groups.get(canonical_path(g, alias_lookup), [])
        p = next((x for x in candidates if x in pred_left), None)
        if p is not None:
            matches.append((g, p))
            gt_left.remove(g)
            pred_left.remove(p)

    gs = {g: alias_lookup.get(norm_key(leaf_key(g)), norm_key(leaf_key(g))) for g in gt_left}
    ps = {p: alias_lookup.get(norm_key(leaf_key(p)), norm_key(leaf_key(p))) for p in pred_left}
    gc, pc = Counter(gs.values()), Counter(ps.values())
    for g in list(gt_left):
        sig = gs[g]
        candidates = [p for p in pred_left if ps[p] == sig]
        if gc[sig] == 1 and pc[sig] == 1 and len(candidates) == 1:
            p = candidates[0]
            matches.append((g, p))
            gt_left.remove(g)
            pred_left.remove(p)

    if mode in {"schemafree", "auto"}:
        scored = []
        for i, g in enumerate(gt_left):
            for j, p in enumerate(pred_left):
                if not has_conflicting_semantics(leaf_key(g), leaf_key(p)):
                    score = SequenceMatcher(None, norm_key(leaf_key(g)),
                                             norm_key(leaf_key(p))).ratio()
                    scored.append((score, i, j))
        used_g, used_p = set(), set()
        for score, i, j in sorted(scored, reverse=True):
            if score < threshold:
                break
            if i not in used_g and j not in used_p:
                matches.append((gt_left[i], pred_left[j]))
                used_g.add(i)
                used_p.add(j)
        gt_left = [x for i, x in enumerate(gt_left) if i not in used_g]
        pred_left = [x for j, x in enumerate(pred_left) if j not in used_p]
    return matches, gt_left, pred_left


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return round(100*p, 2), round(100*r, 2), round(100*f, 2)


def levenshtein(a, b):
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j-1] + 1,
                               previous[j-1] + (ca != cb)))
        previous = current
    return previous[-1]


def canonical_form(flat, alias_lookup):
    items = [{"key": canonical_path(k, alias_lookup), "value": normalize_value(v)}
             for k, v in flat.items()]
    items.sort(key=lambda x: (x["key"], x["value"]))
    return json.dumps(items, ensure_ascii=False, separators=(",", ":"))


def ned_similarity(gt, pred, alias_lookup):
    g, p = canonical_form(gt, alias_lookup), canonical_form(pred, alias_lookup)
    longest = max(len(g), len(p))
    return 100.0 if longest == 0 else round(100 * (1 - levenshtein(g, p) / longest), 2)


def evaluate_one(gt, pred, alias_lookup, mode, threshold):
    matches, missing, extra = align_keys(gt, pred, alias_lookup, mode, threshold)
    tp = fp = fn = 0
    for g, p in matches:
        if normalize_value(gt[g]) == normalize_value(pred[p]):
            tp += 1
        else:
            fp += 1
            fn += 1
    fp += len(extra)
    fn += len(missing)
    precision, recall, f1 = prf(tp, fp, fn)

    # Strict whole-form check: every GT key matched, no extra predicted keys,
    # and every matched value is character-identical after strict_text().
    # ONE DELIBERATE CHANGE from Metric_new.py: a strict-correct form also needs every key matched BY NAME
    # (same canonical key, i.e. exact after normalisation or via the alias table), never by fuzzy similarity.
    # In --metric-mode schema no fuzzy matches exist, so this equals Metric_new.py's strict accuracy there.
    strict = (not missing and not extra and len(gt) == len(pred) and
              all(canonical_path(g, alias_lookup) == canonical_path(p, alias_lookup) for g, p in matches) and
              all(strict_text(gt[g]) == strict_text(pred[p]) for g, p in matches))
    ned = ned_similarity(gt, pred, alias_lookup)
    return {
        "combined_kv_precision_pct": precision,
        "combined_kv_recall_pct": recall,
        "combined_kv_f1_pct": f1,
        "normalized_edit_distance_similarity_pct": ned,
        "strict_form_accuracy_pct": 100.0 if strict else 0.0,
        "strict_form_correct": strict,
        "ground_truth_fields": len(gt),
        "prediction_fields": len(pred),
        "matched_key_pairs": len(matches),
        "missing_keys": len(missing),
        "extra_keys": len(extra),
        "incorrect_values": sum(1 for g, p in matches
                                if normalize_value(gt[g]) != normalize_value(pred[p])),
        "tp": tp, "fp": fp, "fn": fn,
    }


def load_aliases(alias_file):
    aliases = {k: list(v) for k, v in DEFAULT_ALIASES.items()}
    if alias_file:
        custom = json.loads(Path(alias_file).read_text(encoding="utf-8"))
        for key, values in custom.items():
            if not isinstance(values, list):
                raise ValueError(f"Aliases for {key} must be a JSON list.")
            aliases.setdefault(str(key), [])
            aliases[str(key)].extend(str(x) for x in values)
    return aliases
# ---- end of Metric_new.py code ----------------------------------------------------------------------

M = SimpleNamespace(make_alias_lookup=make_alias_lookup, load_aliases=load_aliases, flatten=flatten,
                    evaluate_one=evaluate_one, prf=prf)


def make_metric_config(args) -> dict[str, Any]:
    if not 0 <= args.key_threshold <= 1:
        raise SystemExit("--key-threshold must be between 0 and 1")
    return {"alias_lookup": M.make_alias_lookup(M.load_aliases(args.aliases)),
            "mode": args.metric_mode, "threshold": args.key_threshold}


def score_json(prediction: Optional[Any], ground_truth: Any, mc: dict[str, Any]) -> dict[str, Any]:
    """Per-document scores, exactly as Metric_new.evaluate_one computes them."""
    valid = isinstance(prediction, (dict, list))
    gt = M.flatten(ground_truth)
    pred = M.flatten(prediction) if valid else {}
    row = M.evaluate_one(gt, pred, mc["alias_lookup"], mc["mode"], mc["threshold"])
    row["valid_json"] = int(valid)
    return row


def summarize_by_dataset(scores: list[dict[str, Any]], datasets: list[str], min_docs: int) -> dict[str, Any]:
    """Overall result plus one result per dataset, each with its document count and a reliability label."""
    result = {"overall": summarize(scores)}
    for ds in sorted(set(datasets)):
        part = [s for s, d in zip(scores, datasets) if d == ds]
        entry = summarize(part)
        entry["reliability"] = "PRELIMINARY (fewer than %d documents)" % min_docs if len(part) < min_docs else "ok"
        result[ds] = entry
    return result


def summarize(scores: list[dict[str, Any]]) -> dict[str, Any]:
    """Same aggregation as Metric_new.evaluate_folder: P/R/F1 micro over all key-value pairs, FormSim macro over
    documents, Strict Form Accuracy = share of strict-correct documents (all in %)."""
    n = len(scores)
    tp, fp, fn = (sum(s[k] for s in scores) for k in ("tp", "fp", "fn"))
    precision, recall, f1 = M.prf(tp, fp, fn)
    strict = sum(int(s["strict_form_correct"]) for s in scores)
    return {
        "documents": n,
        "combined_kv_precision_pct": precision,
        "combined_kv_recall_pct": recall,
        "combined_kv_f1_pct": f1,
        "normalized_edit_distance_similarity_pct":
            round(sum(s["normalized_edit_distance_similarity_pct"] for s in scores) / n, 2) if n else 0.0,
        "strict_form_accuracy_pct": round(100 * strict / n, 2) if n else 0.0,
        "strict_form_correct_documents": strict,
        "combined_kv_tp": tp, "combined_kv_fp": fp, "combined_kv_fn": fn,
        "json_valid_rate_pct": round(100 * sum(s["valid_json"] for s in scores) / n, 2) if n else 0.0,
    }


# =============================================================================
# 4. MAIN
# =============================================================================
def run(args) -> None:
    external = bool(args.external_predictions)
    if external:
        if args.adapter or args.images or not args.data or not args.out:
            raise SystemExit("--external-predictions needs --data, --split (with --run-dir for validation/test) and "
                             "--out, and no --adapter / --images.")
        settings = {"max_pixels": None, "min_pixels": None, "revision": None}
        model_name = args.external_name
        out_dir = Path(args.out)
    else:
        settings = resolve_settings(args)
        model_name = T.SPEC.alias
        out_dir = Path(args.out) if args.out else (
            Path(args.adapter).resolve().parent / "inference" if args.adapter else Path(f"runs/base_{T.SPEC.family}"))
    tag = args.split if not args.images else "single"
    if not args.overwrite and ((out_dir / f"eval_summary_{tag}.json").exists()
                               or (out_dir / f"predictions_{tag}.jsonl").exists()):
        raise SystemExit(f"{out_dir} already holds results for '{tag}'. Use a new --out folder (or --overwrite).")
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_run_logging(out_dir, "infer")
    enforce_lock(args)
    records = build_records(args)
    metric_cfg = make_metric_config(args)
    best_report = verify_best_adapter(args.adapter) if args.adapter else None
    missing_external: list[str] = []

    if external:
        ext_texts, missing_external = load_external_predictions(args.external_predictions, records)
        ext_by_id = {r["id"]: t for r, t in zip(records, ext_texts)}
        variants = [args.external_name]
        LOG.info("%s (external predictions from %s): %d document(s)", model_name, args.external_predictions,
                 len(records))
    else:
        from peft import PeftModel
        LOG.info("%s: %d document(s), batch size %d, revision %s", model_name, len(records), args.batch_size,
                 resolve_model_revision(settings["revision"]))
        processor = load_processor(settings["max_pixels"], settings["min_pixels"], settings["revision"])
        base_model = load_base_model(settings["revision"])
        model = PeftModel.from_pretrained(base_model, args.adapter, is_trainable=False) if args.adapter else base_model
        model.eval()
        if getattr(model, "config", None) is not None:
            model.config.use_cache = True                 # training disabled the KV cache; generation needs it
            text_config = getattr(model.config, "text_config", None)
            if text_config is not None and hasattr(text_config, "use_cache"):
                text_config.use_cache = True
        variants = ["finetuned" if args.adapter else "base"]
        if args.compare_base and args.adapter:
            variants.append("base")       # same records, prompt and preprocessing; adapter switched off
    pred_path = out_dir / f"predictions_{tag}.jsonl"
    scores: dict[str, list[dict[str, int]]] = {v: [] for v in variants}
    scored_datasets: list[str] = []
    with pred_path.open("w", encoding="utf-8") as f:
        for start in range(0, len(records), args.batch_size):
            chunk = records[start:start + args.batch_size]
            outputs: dict[str, list[str]] = {}
            for variant in variants:
                if external:
                    outputs[variant] = [ext_by_id[r["id"]] or "" for r in chunk]
                    continue
                ctx = model.disable_adapter() if (variant == "base" and args.adapter) else nullcontext()
                with ctx:
                    outputs[variant] = generate_with_backoff(model, processor, chunk, args.max_new_tokens)
            for i, record in enumerate(chunk):
                item: dict[str, Any] = {"id": record["id"], "images": record["images"]}
                for key in ("dataset", "group_id", "is_augmented", "source_id"):
                    if key in record:
                        item[key] = record[key]
                if record["target"] is not None:
                    item["target"] = record["target"]
                    scored_datasets.append(record.get("dataset", "unknown"))
                for variant in variants:
                    raw = outputs[variant][i]
                    parsed = parse_json_answer(raw)
                    entry: dict[str, Any] = {"raw": raw, "parsed": parsed}
                    if record["target"] is not None:
                        entry["score"] = score_json(parsed, record["target"], metric_cfg)
                        scores[variant].append(entry["score"])
                    item[variant] = entry
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                if args.images:
                    first = item[variants[0]]
                    print(json.dumps(first["parsed"] if first["parsed"] is not None
                                     else {"_unparsed_output": first["raw"]}, ensure_ascii=False, indent=2))
            LOG.info("[%d/%d] done", min(start + args.batch_size, len(records)), len(records))

    LOG.info("Predictions: %s", pred_path)
    if not args.allow_unlabeled and not args.images:
        for variant, s in scores.items():
            if len(s) != len(records):
                raise SystemExit(f"{variant}: scored {len(s)} of {len(records)} documents; refusing to write an "
                                 "incomplete summary.")
    if any(scores.values()):
        summary = {v: summarize_by_dataset(s, scored_datasets, args.min_docs) for v, s in scores.items() if s}
        summary["_meta"] = {"model": model_name, "adapter": args.adapter,
                            "external_predictions": args.external_predictions,
                            "missing_external_predictions": missing_external,
                            "best_checkpoint_check": best_report,
                            "metric_definitions": {
                                "combined_kv_*": "key-value pairs: TP = key matched and value equal after "
                                                 "normalisation; wrong value = FP + FN; micro-averaged",
                                "normalized_edit_distance_similarity_pct": "FormSim: custom whole-form JSON "
                                    "similarity = 1 - Levenshtein(normalised, sorted, serialised key-value JSON of GT "
                                    "vs prediction) / max length; keys and JSON structure count; macro-averaged",
                                "strict_form_accuracy_pct": "share of documents with every key matched by name and "
                                                            "every value equal (NFKC, lowercase, whitespace)"}, "data": args.data, "split": args.split,
                            "max_pixels": settings["max_pixels"], "revision": settings["revision"],
                            "metrics": "Metric_new.py definitions (built into infer.py)", "metric_mode": args.metric_mode,
                            "key_threshold": args.key_threshold, "aliases": args.aliases,
                            "strict_rule": "all GT keys matched by name (canonical/alias), no extra keys, values equal "
                                           "after NFKC + lowercase + whitespace collapsing",
                            "documents_selected": len(records), "documents_scored": len(scored_datasets),
                            "manifest_sha256": file_sha256(Path(args.data)) if args.data else None,
                            "max_new_tokens": args.max_new_tokens}
        summary_path = out_dir / f"eval_summary_{tag}.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        for variant, per in summary.items():
            if variant.startswith("_"):
                continue
            print(f"\n{variant}:  (all values in %; FormSim = whole-form JSON similarity)\n  {'dataset':9s} "
                  f"{'docs':>5s} {'KV_prec':>8s} {'KV_rec':>8s} {'KV_F1':>8s} {'FormSim':>8s} {'strict':>8s} "
                  f"{'json_ok':>8s}")
            for ds, e in per.items():
                flag = "" if ds == "overall" or e["reliability"] == "ok" else "  PRELIMINARY"
                print(f"  {ds:9s} {e['documents']:5d} {e['combined_kv_precision_pct']:8.2f} "
                      f"{e['combined_kv_recall_pct']:8.2f} {e['combined_kv_f1_pct']:8.2f} "
                      f"{e['normalized_edit_distance_similarity_pct']:8.2f} {e['strict_form_accuracy_pct']:8.2f} "
                      f"{e['json_valid_rate_pct']:8.2f}{flag}")
        LOG.info("Summary: %s", summary_path)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--adapter", default=None, help="final_adapter directory; omit to run the base model")
    p.add_argument("--model", default=None, choices=sorted(MODEL_REGISTRY), help="Base model (default: from adapter)")
    p.add_argument("--revision", default=None, help="Default: as trained")
    p.add_argument("--lock", default=None, help="Lock file from `freeze`")
    p.add_argument("--max-pixels", type=int, default=None, help="Default: as trained")
    p.add_argument("--min-pixels", type=int, default=None, help="Default: as trained")
    p.add_argument("--images", nargs="+", default=None, help="Page images of ONE document, in page order")
    p.add_argument("--doc-id", default="document")
    p.add_argument("--data", default=None, help="JSONL manifest")
    p.add_argument("--schema", default=None, help="Schema JSON (required with --images; fallback for --data)")
    p.add_argument("--split", default="validation", choices=["test", "validation", "train", "all"])
    p.add_argument("--run-dir", default=None, help="Folder with splits.json (default: parent of --adapter)")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=1, help="Documents per generate() call")
    p.add_argument("--max-new-tokens", type=int, default=2048,
                   help="Generation limit; the longest training answer is ~1,300 tokens (generation stops at the end "
                        "token, so a high limit costs nothing for short answers)")
    p.add_argument("--compare-base", action="store_true", help="Also run with the adapter disabled")
    p.add_argument("--min-docs", type=int, default=30, help="Per-dataset results below this many documents are "
                   "labelled PRELIMINARY")
    p.add_argument("--metric-mode", default="schema", choices=["schema", "schemafree", "auto"],
                   help="Key alignment: schema (default) = exact key name or alias only; auto/schemafree also "
                        "fuzzy (lexical) key matching. Use the same mode for every model you compare.")
    p.add_argument("--key-threshold", type=float, default=0.80, help="Lexical key-match threshold")
    p.add_argument("--aliases", default=None, help="Optional custom key-alias JSON (added to the built-in aliases)")
    p.add_argument("--external-predictions", default=None,
                   help="Score another model's saved outputs (folder of <image-name>.json, or JSONL with id + "
                        "prediction) on the same documents and rules, without loading a model")
    p.add_argument("--external-name", default="external", help="Name for --external-predictions, e.g. gemini")
    p.add_argument("--overwrite", action="store_true", help="Allow writing into a folder that already has results")
    p.add_argument("--allow-unlabeled", action="store_true",
                   help="Allow records without ground truth or dataset label (predictions only; not for scoring)")
    p.add_argument("--out", default=None, help="Output folder (default: <run>/inference)")
    return p


def main(argv: Optional[list[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be > 0")
    try:
        run(args)
    except SystemExit:
        raise
    except BaseException as exc:
        LOG.exception("FAILED: %s", exc)
        raise


if __name__ == "__main__":
    main()
