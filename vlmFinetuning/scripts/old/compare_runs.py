#!/usr/bin/env python3
"""Compare runs (models x LoRA recipes) side by side.

  python scripts/compare_runs.py --plans runs/inspect_qwen25vl_A runs/inspect_qwen25vl_B runs/inspect_qwen25vl_C
      before training: modules and LoRA parameters per region (from lora_target_summary.json)
  python scripts/compare_runs.py runs/qwen25vl_A runs/qwen25vl_C runs/nuextract3_A runs/glm46v_A ...
      after training: trainable parameters, peak VRAM, best validation loss, and field accuracy per dataset on
      validation, final test and the legacy benchmark: Combined KV F1, NED similarity and Strict Form Accuracy
      (in %, from Metric_new.py) per dataset
      (document counts in brackets; * = fewer than 30 = preliminary)
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def show(rows: list[dict], cols: list[str]) -> None:
    w = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    print("  ".join(c.ljust(w[c]) for c in cols))
    print("  ".join("-" * w[c] for c in cols))
    for r in rows:
        print("  ".join(str(r.get(c, "")).ljust(w[c]) for c in cols))


def plans(dirs: list[Path]) -> None:
    rows = []
    for d in dirs:
        s = load(d / "lora_target_summary.json")
        if not s:
            print(f"skip {d}: no lora_target_summary.json")
            continue
        row = {"plan": d.name, "model": s.get("model"), "regions": "+".join(s.get("selected_regions", [])),
               "vision_last_n": s.get("vision_last_n"), "modules": s["total_planned_modules"],
               "lora_params_M": round(s["total_lora_params"] / 1e6, 2)}
        for reg in ("vision", "projector", "language"):
            row[f"{reg}_modules"] = s["regions"].get(reg, {}).get("planned_modules", 0)
        rows.append(row)
    if rows:
        show(rows, list(rows[0]))


def cell(summary: dict, variant: str, ds: str) -> str:
    e = (summary or {}).get(variant, {}).get(ds)
    if not e:
        return ""
    star = "*" if e["documents"] < 30 else ""
    if "combined_kv_f1_pct" not in e:          # summary written by the old infer.py
        return f"old metrics ({e['documents']}{star})"
    return (f"F1 {e['combined_kv_f1_pct']:.2f} NED {e['normalized_edit_distance_similarity_pct']:.2f} "
            f"strict {e['strict_form_accuracy_pct']:.2f} ({e['documents']}{star})")


def results(dirs: list[Path], out_csv: Path) -> None:
    rows = []
    for d in dirs:
        meta = load(d / "run_metadata.json") or {}
        status = load(d / "run_status.json") or {}
        att = meta.get("lora_attachment_check") or {}
        row = {"run": d.name, "status": status.get("status", "?"), "model": meta.get("model_alias", ""),
               "regions": "+".join(meta.get("regions", [])), "vision_last_n": meta.get("vision_last_n", ""),
               "mix": meta.get("dataset_mix", {}).get("sampled_share") or "natural" if meta else "",
               "trainable_M": round(att.get("trainable_params", 0) / 1e6, 2) if att else "",
               "peak_vram_GiB": meta.get("peak_reserved_vram_gib", ""),
               "best_val_loss": round(meta["best_eval_loss"], 4) if meta.get("best_eval_loss") is not None else "",
               "steps": meta.get("global_step", "")}
        for split, path in (("val", d / "validation" / "eval_summary_validation.json"),
                            ("test", d / "final_test" / "eval_summary_test.json")):
            s = load(path)
            for ds in ("overall", "forms", "cheques", "funsd"):
                row[f"{split}_{ds}"] = cell(s, "finetuned", ds)
        for ds in ("forms", "cheques", "funsd"):
            s = load(d / f"legacy_{ds}" / "eval_summary_all.json")
            row[f"legacy_{ds}"] = cell(s, "finetuned", ds)
            if s and "base" in s:
                row[f"legacy_{ds}_base"] = cell(s, "base", ds)
        rows.append(row)
    if not rows:
        return
    cols = list(dict.fromkeys(k for r in rows for k in r))
    show(rows, cols)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"\nCombined KV F1 / NED similarity / Strict Form Accuracy in % (documents; * = under 30, preliminary). Wrote {out_csv}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="*", type=Path)
    p.add_argument("--plans", nargs="*", type=Path, default=None)
    p.add_argument("--csv", type=Path, default=Path("runs/comparison.csv"))
    a = p.parse_args()
    if a.plans:
        plans(a.plans)
    if a.runs:
        a.csv.parent.mkdir(parents=True, exist_ok=True)
        results(a.runs, a.csv)


if __name__ == "__main__":
    main()
