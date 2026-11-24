#!/usr/bin/env python3
"""Independent leakage check of prepared/ against your dataset/ folders (written separately from prepare_data.py).

Checks:
  1. every image and every document (group_id) is in exactly one split;
  2. every image of a split really lies in that split's folder (dataset/<ds>/<split>/images);
  3. validation and test contain no tilted file (judged by file NAME);
  4. every tilt used in training belongs to a training document, and every training document brings all its tilts;
  5. no benchmark file's content (bytes, originals and tilts) is used in any split;
  6. no FUNSD benchmark source page (funsd/benchmark/source_pages.txt) or its tilt is used.
Exit code 0 = passed.     Usage: python scripts/verify_split.py [--dataset dataset] [--prepared prepared]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
TILT = re.compile(r"(?:__tilt\d+|\.\d+)$")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="dataset")
    ap.add_argument("--prepared", default="prepared")
    ap.add_argument("--splits", default=None, help="Default: <prepared>/splits/splits.json")
    a = ap.parse_args()
    ds_root, prep = Path(a.dataset).resolve(), Path(a.prepared)
    man = {r["id"]: r for r in map(json.loads, (prep / "manifest.jsonl").read_text(encoding="utf-8").splitlines())}
    sp = json.loads(Path(a.splits or prep / "splits" / "splits.json").read_text(encoding="utf-8"))
    img = lambda rid: (prep / man[rid]["images"][0]).resolve()
    stem = lambda p: p.with_suffix("").name
    fails: list[str] = []

    seen, gsplit = {}, defaultdict(set)
    for s in ("train", "validation", "test"):
        for rid in sp.get(s, []):
            if rid in seen:
                fails.append(f"1: {rid} in {seen[rid]} and {s}")
            seen[rid] = s
            gsplit[man[rid]["group_id"]].add(s)
            folder = ds_root / man[rid]["dataset"] / s / "images"
            if folder not in img(rid).parents:
                fails.append(f"2: {rid} is used as {s} but its file is not in {folder}")
    fails += [f"1: document {g} in {sorted(v)}" for g, v in gsplit.items() if len(v) > 1]

    for s in ("validation", "test"):
        fails += [f"3: tilted file {img(r).name} used in {s}" for r in sp.get(s, []) if TILT.search(stem(img(r)))]

    train_imgs = {img(r) for r in sp.get("train", [])}
    for ds_dir in sorted(p for p in ds_root.iterdir() if p.is_dir()):
        tdir = ds_dir / "train" / "images"
        if not tdir.is_dir():
            continue
        files = [p for p in tdir.rglob("*") if p.suffix.lower() in IMAGE_EXT]
        originals = {TILT.sub("", stem(p)) for p in files if not TILT.search(stem(p))}
        for p in files:
            if TILT.search(stem(p)):
                if TILT.sub("", stem(p)) not in originals:
                    fails.append(f"4: {ds_dir.name}/train has tilt {p.name} without its original")
                elif p.resolve() not in train_imgs:
                    fails.append(f"4: tilt {ds_dir.name}/train/{p.name} of a training document is not used")

    used_hash = {hashlib.sha256(p.read_bytes()).hexdigest(): p for p in (img(r) for r in seen)}
    for b in sorted(ds_root.glob("*/benchmark/images/*")):
        if b.suffix.lower() in IMAGE_EXT:
            h = hashlib.sha256(b.read_bytes()).hexdigest()
            if h in used_hash:
                fails.append(f"5: benchmark file {b.parent.parent.parent.name}/benchmark/{b.name} is used "
                             f"as {used_hash[h].relative_to(ds_root)}")

    spf = ds_root / "funsd" / "benchmark" / "source_pages.txt"
    if spf.is_file():
        pages = {Path(l.split("#", 1)[0].strip()).with_suffix("").name for l in spf.read_text(encoding="utf-8").splitlines()
                 if l.split("#", 1)[0].strip()}
        for r in seen:
            if man[r]["dataset"] == "funsd" and TILT.sub("", stem(img(r))) in pages:
                fails.append(f"6: FUNSD benchmark source page used: {img(r).name}")

    print(f"Checked {len(seen)} images in {len(gsplit)} documents (train {len(sp.get('train', []))}, "
          f"validation {len(sp.get('validation', []))}, test {len(sp.get('test', []))}).")
    if fails:
        print(f"LEAKAGE CHECK FAILED: {len(fails)} problem(s)")
        for f in sorted(set(fails))[:30]:
            print("  " + f)
        sys.exit(1)
    print("LEAKAGE CHECK PASSED: each document in one folder/split, validation/test originals only, all training "
          "tilts used, no benchmark document or FUNSD source page used.")


if __name__ == "__main__":
    main()
