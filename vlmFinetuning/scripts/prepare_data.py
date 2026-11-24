#!/usr/bin/env python3
"""Read the MANUAL split from dataset/ and build the training manifest, for all three models.

You choose the split by putting the files in folders; this script does not choose anything. It reads:

  dataset/<forms|cheques|funsd>/train/       images/ + gt_json/   originals AND their tilts
  dataset/<forms|cheques|funsd>/validation/  images/ + gt_json/   originals only
  dataset/<forms|cheques|funsd>/test/        images/ + gt_json/   originals only
  dataset/<forms|cheques|funsd>/benchmark/   images/ + gt_json/   legacy benchmark, originals only
  dataset/funsd/benchmark/source_pages.txt   the FUNSD pages the benchmark crops were cut from

File names: original <name>.png, tilts <name>.1.jpg, <name>.2.jpg (also <name>__tilt1.png); ground truth with the
same name in gt_json/ (<name>.json, <name>.1.json, ...). A tilt without its own JSON uses its original's.

It STOPS (writes nothing) if the folders break a rule:
  - the same document (name) in two folders, or the same image (identical file content) in two folders;
  - a tilt whose original is not in the same folder (e.g. a tilt of a test form left in train/);
  - an original without a ground-truth JSON;
  - a FUNSD benchmark source page (source_pages.txt) in train/, validation/ or test/.
It WARNS (and ignores the files) for tilts in validation/, test/ or benchmark/: those sets use originals only.

Output (prepared/)
  manifest.jsonl            every train/validation/test image with dataset, group_id, is_augmented, benchmark_role
  splits/splits.json        image IDs per split + manifest fingerprint (train.py checks it)
  splits/group_assignments.json   document -> split, as found in your folders
  smoke/splits.json         a small subset of the same split for the smoke test
  legacy_<dataset>.jsonl    the benchmark originals, for reference evaluation
  reports/                  split_report.txt (counts vs config/expected_counts.json), near_duplicates.csv,
                            unused_tilts.csv, tilt_label_differences.csv

Usage:  python scripts/prepare_data.py            (run again whenever files in dataset/ change)
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

DATASETS = ("forms", "cheques", "funsd")
FOLDERS = ("train", "validation", "test", "benchmark")
SPLITS = ("train", "validation", "test")
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
TILT_RE = re.compile(r"(?:__tilt\d+|\.\d+)$")


def log(msg: str) -> None:
    print(msg, flush=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def key_of(rel: Path) -> str:
    """Relative path without the image extension: 'form_001', 'form_001.1'."""
    return rel.with_suffix("").as_posix() if rel.suffix.lower() in IMAGE_EXT | {".json"} else rel.as_posix()


def base_key(key: str) -> Optional[str]:
    """'form_001.1' -> 'form_001'; None if the key is an original."""
    head, _, last = key.rpartition("/")
    m = TILT_RE.search(last)
    return None if not m else (head + "/" if head else "") + last[:m.start()]


def read_label(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, dict) and isinstance(obj.get("target"), dict):
        return {"target": obj["target"], "schema": obj.get("schema")}
    if isinstance(obj, dict):
        return {"target": obj, "schema": None}
    raise SystemExit(f"{path}: a ground truth must be a JSON object (field -> value)")


def dhash(path: Path, size: int = 8) -> int:
    from PIL import Image
    with Image.open(path) as im:
        g = im.convert("L").resize((size + 1, size), Image.BILINEAR)
    px = list(g.tobytes())
    bits = 0
    for r in range(size):
        for c in range(size):
            bits = (bits << 1) | (px[r * (size + 1) + c] > px[r * (size + 1) + c + 1])
    return bits


class Folder:
    """One dataset/<ds>/<folder>/ with its originals, tilts and ground truths."""

    def __init__(self, ds: str, name: str, root: Path):
        self.ds, self.name, self.root = ds, name, root
        self.originals: dict[str, Path] = {}
        self.tilts: dict[str, list[Path]] = defaultdict(list)
        self.labels: dict[str, dict[str, Any]] = {}
        self.orphan_tilts: list[str] = []
        self.missing_labels: list[str] = []
        self.hash: dict[Path, str] = {}

    def scan(self) -> None:
        img_dir, gt_dir = self.root / "images", self.root / "gt_json"
        if not img_dir.is_dir():
            return
        files = sorted(p for p in img_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXT)
        for p in files:
            k = key_of(p.relative_to(img_dir))
            b = base_key(k)
            if b is None:
                self.originals[k] = p
            else:
                self.tilts[b].append(p)
            self.hash[p] = sha256_file(p)
        self.orphan_tilts = sorted(f"{b} ({len(v)} tilt(s))" for b, v in self.tilts.items() if b not in self.originals)
        if gt_dir.is_dir():
            for j in sorted(gt_dir.rglob("*.json")):
                self.labels[key_of(j.relative_to(gt_dir))] = read_label(j)
        self.missing_labels = sorted(k for k in self.originals if k not in self.labels)

    def n_images(self) -> int:
        return len(self.originals) + sum(len(v) for v in self.tilts.values())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="dataset", help="the folder you pasted the data into")
    ap.add_argument("--out", default="prepared")
    ap.add_argument("--expect", default="config/expected_counts.json", help="numbers to verify ('' to skip)")
    ap.add_argument("--strict", action="store_true", help="also stop when a count differs from --expect")
    ap.add_argument("--smoke-train-docs", type=int, default=6, help="training documents per dataset in the smoke split")
    ap.add_argument("--near-dup-bits", type=int, default=3, help="dHash distance reported as a possible same document")
    a = ap.parse_args()
    root, out = Path(a.dataset).resolve(), Path(a.out).resolve()
    if not root.is_dir():
        raise SystemExit(f"{root} not found: paste your data into dataset/<forms|cheques|funsd>/<train|validation|test|benchmark>/")

    # ---------- 1. read every folder ----------
    F: dict[tuple[str, str], Folder] = {}
    names = [d for d in DATASETS if (root / d).is_dir()]
    for ds in names:
        for fo in FOLDERS:
            f = Folder(ds, fo, root / ds / fo)
            f.scan()
            F[(ds, fo)] = f
        log(f"[{ds}] " + " | ".join(f"{fo} {len(F[(ds, fo)].originals)} docs / {F[(ds, fo)].n_images()} images" for fo in FOLDERS))

    # ---------- 2. rule checks ----------
    problems: list[str] = []
    warnings: list[str] = []
    unused: list[tuple[str, str, str]] = []
    for ds in names:
        where: dict[str, list[str]] = defaultdict(list)
        for fo in FOLDERS:
            f = F[(ds, fo)]
            for k in f.originals:
                where[k].append(fo)
            problems += [f"[{ds}/{fo}] tilt without its original in the same folder: {t}" for t in f.orphan_tilts]
            problems += [f"[{ds}/{fo}] original without ground truth: gt_json/{k}.json missing" for k in f.missing_labels]
            if fo != "train":
                for b, ts in f.tilts.items():
                    for t in ts:
                        unused.append((ds, fo, t.name))
                if f.tilts:
                    warnings.append(f"[{ds}/{fo}] {sum(len(v) for v in f.tilts.values())} tilted files ignored "
                                    f"({fo} uses originals only)")
        problems += [f"[{ds}] document {k!r} is in more than one folder: {v}" for k, v in sorted(where.items()) if len(v) > 1]
        content: dict[str, list[str]] = defaultdict(list)
        for fo in FOLDERS:
            f = F[(ds, fo)]
            used = list(f.originals.values()) + ([t for ts in f.tilts.values() for t in ts] if fo == "train" else [])
            for p in used:
                content[f.hash[p]].append(f"{fo}/{p.name}")
        for paths in content.values():
            folders = {p.split("/")[0] for p in paths}
            if len(folders) > 1:
                problems.append(f"[{ds}] the same image is in different folders: {sorted(paths)}")
        if ds == "funsd":
            sp = root / "funsd" / "benchmark" / "source_pages.txt"
            pages = []
            if sp.is_file():
                pages = [key_of(Path(l.split("#", 1)[0].strip())) for l in sp.read_text(encoding="utf-8").splitlines()
                         if l.split("#", 1)[0].strip()]
            if not pages and F[("funsd", "benchmark")].originals:
                warnings.append("[funsd] benchmark/source_pages.txt is empty: cannot check that the 5 pages the "
                                "benchmark crops came from are not in train/validation/test")
            for pg in pages:
                for fo in SPLITS:
                    f = F[("funsd", fo)]
                    if pg in f.originals or pg in f.tilts:
                        problems.append(f"[funsd] benchmark source page {pg!r} is in {fo}/ (it must not be used)")
    for w in warnings:
        log("  WARNING " + w)
    if problems:
        for p in problems:
            log("  PROBLEM " + p)
        raise SystemExit(f"\n{len(problems)} problem(s) in dataset/: fix the folders and run again (nothing was written).")

    # ---------- 3. records ----------
    out.mkdir(parents=True, exist_ok=True)
    for sub in ("splits", "smoke", "reports"):
        (out / sub).mkdir(exist_ok=True)
    rel = lambda p: os.path.relpath(p, out).replace(os.sep, "/")
    records: list[dict[str, Any]] = []
    split_ids: dict[str, list[str]] = {s: [] for s in SPLITS}
    groups: dict[str, dict[str, str]] = {}
    label_diffs: list[tuple[str, str, str]] = []
    counts: dict[str, Counter] = {}
    for ds in names:
        c = counts.setdefault(ds, Counter())
        groups[ds] = {}
        for fo in SPLITS:
            f = F[(ds, fo)]
            for k in sorted(f.originals):
                gid = f"{ds}:{k}"
                groups[ds][gid] = fo
                c[f"{fo}_docs"] += 1
                entries = [(f.originals[k], False)] + ([(t, True) for t in sorted(f.tilts.get(k, []))] if fo == "train" else [])
                for path, aug in entries:
                    ik = key_of(path.relative_to(f.root / "images"))
                    lab = f.labels.get(ik) or f.labels[k]
                    if aug and ik in f.labels and f.labels[ik]["target"] != f.labels[k]["target"]:
                        label_diffs.append((ds, k, ik))
                    rid = f"{ds}/{fo}/{ik}"
                    rec = {"id": rid, "images": [rel(path)], "target": lab["target"], "dataset": ds, "group_id": gid,
                           "is_augmented": aug, "benchmark_role": "none", "source_id": f"{ds}/{k}"}
                    if lab.get("schema"):
                        rec["schema"] = lab["schema"]
                    records.append(rec)
                    split_ids[fo].append(rid)
                    c[f"{fo}_images"] += 1
        tr = F[(ds, "train")]
        hist = Counter(len(tr.tilts.get(k, [])) for k in tr.originals)
        c["tilts_per_train_doc"] = max(hist, key=hist.get) if hist else 0
        if len(hist) > 1:
            warnings.append(f"[{ds}] training documents do not all have the same number of tilts: {dict(sorted(hist.items()))}")
        b = F[(ds, "benchmark")]
        c["benchmark_docs"] = len(b.originals)
        for k in sorted(b.originals):
            groups[ds][f"{ds}:benchmark/{k}"] = "benchmark"

    manifest = out / "manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    msha = sha256_file(manifest)
    for ds in names:
        b = F[(ds, "benchmark")]
        with (out / f"legacy_{ds}.jsonl").open("w", encoding="utf-8") as fh:
            for k in sorted(b.originals):
                rec = {"id": f"legacy_{ds}/{k}", "images": [rel(b.originals[k])], "dataset": ds,
                       "group_id": f"{ds}:benchmark/{k}", "is_augmented": False, "benchmark_role": "legacy",
                       "source_id": f"{ds}/benchmark/{k}"}
                if k in b.labels:
                    rec["target"] = b.labels[k]["target"]
                    if b.labels[k].get("schema"):
                        rec["schema"] = b.labels[k]["schema"]
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    (out / "splits" / "group_assignments.json").write_text(json.dumps(
        {"_meta": {"source": "manual folders in " + str(root)}, "groups": groups}, indent=1), encoding="utf-8")
    meta = {"manifest_sha256": msha, "created_by": "prepare_data.py (manual folder split)",
            "assigned_groups": sum(len(g) for g in groups.values())}
    (out / "splits" / "splits.json").write_text(json.dumps({**split_ids, "_meta": meta}, indent=1), encoding="utf-8")
    by_id = {r["id"]: r for r in records}
    smoke = {s: [] for s in SPLITS}
    for ds in names:
        for fo, n in (("train", a.smoke_train_docs), ("validation", 2), ("test", 1)):
            gids = sorted({by_id[i]["group_id"] for i in split_ids[fo] if by_id[i]["dataset"] == ds})[:n]
            smoke[fo] += [i for i in split_ids[fo] if by_id[i]["group_id"] in gids]
    (out / "smoke" / "splits.json").write_text(json.dumps({**smoke, "_meta": {**meta, "subset": "smoke"}}, indent=1),
                                               encoding="utf-8")

    # ---------- 4. reports ----------
    near = []
    rd = out / "reports"
    try:
        for ds in names:
            origs = [(fo, k, p) for fo in FOLDERS for k, p in F[(ds, fo)].originals.items()]
            hs = [(fo, k, dhash(p)) for fo, k, p in origs]
            for i, (fa, ka, ha) in enumerate(hs):
                for fb, kb, hb in hs[i + 1:]:
                    d = bin(ha ^ hb).count("1")
                    if d <= a.near_dup_bits and fa != fb:
                        near.append((ds, f"{fa}/{ka}", f"{fb}/{kb}", d))
        with (rd / "near_duplicates.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh); w.writerow(["dataset", "document_a", "document_b", "dhash_distance"]); w.writerows(near)
    except (ImportError, ModuleNotFoundError):
        log("  [info] PIL not installed; preserved/skipped dHash near-duplicate scan")
        if not (rd / "near_duplicates.csv").is_file():
            with (rd / "near_duplicates.csv").open("w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh); w.writerow(["dataset", "document_a", "document_b", "dhash_distance"])
    with (rd / "unused_tilts.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["dataset", "folder", "file"]); w.writerows(unused)
    with (rd / "tilt_label_differences.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["dataset", "original", "tilted_copy"]); w.writerows(label_diffs)

    expected = json.loads(Path(a.expect).read_text()) if a.expect and Path(a.expect).is_file() else {}
    rows = ["train_docs", "tilts_per_train_doc", "train_images", "validation_docs", "test_docs", "benchmark_docs"]
    lines = ["Split report - MANUAL split read from " + str(root), f"manifest {manifest} sha256 {msha[:16]}", ""]
    head = f"{'':22s}" + "".join(f"{n:>16s}" for n in names) + f"{'total':>9s}"
    lines += [head, "-" * len(head)]
    diffs = []
    for row in rows:
        cells, tot = [], 0
        for ds in names:
            v = counts[ds].get(row, 0)
            tot += v
            e = expected.get(ds, {}).get(row)
            if e is not None and e != v:
                diffs.append(f"{ds}.{row}: found {v}, expected {e}")
            cells.append(f"{str(v) + ('' if e is None else (' ok' if e == v else f' !{e}')):>16s}")
        lines.append(f"{row:22s}" + "".join(cells) + (f"{tot:>9d}" if row != "tilts_per_train_doc" else ""))
    ntr = len(split_ids["train"])
    if ntr:
        lines += ["", "Training mix by images: " + ", ".join(f"{ds} {counts[ds]['train_images'] / ntr:.1%}" for ds in names)]
    small = [f"{ds} ({counts[ds]['test_docs']})" for ds in names if counts[ds]["test_docs"] < 30]
    if small:
        lines.append("Test sets below 30 documents (report per-dataset results as PRELIMINARY): " + ", ".join(small))
    lines += ["WARNING " + w for w in warnings]
    if label_diffs:
        lines.append(f"{len(label_diffs)} tilts have a ground truth different from their original "
                     "(each image uses its own; see reports/tilt_label_differences.csv)")
    if near:
        lines.append(f"{len(near)} pairs of documents in DIFFERENT folders look almost identical "
                     "(reports/near_duplicates.csv): if they are the same filled document, move them into one folder.")
    lines += (["", "DIFFERENCES from " + a.expect + ":"] + ["  " + d for d in diffs]) if diffs else \
             (["", "All counts match " + a.expect + "."] if expected else [])
    report = "\n".join(lines)
    (rd / "split_report.txt").write_text(report + "\n", encoding="utf-8")
    log("\n" + report)
    log(f"\nWrote {manifest}, {out / 'splits' / 'splits.json'}, {out / 'smoke' / 'splits.json'}, legacy_<dataset>.jsonl, reports/")
    if diffs and a.strict:
        sys.exit(1)


if __name__ == "__main__":
    main()
