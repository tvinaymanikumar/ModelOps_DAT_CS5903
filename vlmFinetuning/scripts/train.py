#!/usr/bin/env python3
"""LoRA fine-tuning for document image -> JSON field extraction, for three models in one script.

  --model qwen2.5-vl-7b    Qwen/Qwen2.5-VL-7B-Instruct
  --model nuextract3       numind/NuExtract3
  --model glm-4.6v-flash   zai-org/GLM-4.6V-Flash   (trained with thinking OFF)

The base model is loaded in BF16 (FP16 on GPUs without BF16) and stays frozen; only LoRA matrices train.
Data comes from scripts/prepare_data.py: prepared/manifest.jsonl + prepared/splits/splits.json, read from YOUR
folders dataset/<ds>/{train,validation,test}. A document and its tilts are one group_id, always in the same split.

Commands
  inspect        Load the model (or only its structure with --no-weights), list every Linear layer by region,
                 and write the LoRA target plan: lora_target_plan.csv + lora_target_summary.json (plan only).
  preview-batch  Load only the processor and print exactly what one training batch contains (no GPU needed).
  train          Train one adapter. Checks before the first step that PEFT wrapped exactly the planned modules,
                 on the first backward pass that every region gets finite, nonzero gradients, and that the split
                 has no leakage between source documents. Checkpoints every --save-steps; --resume auto continues.
  freeze         Write a lock file with the exact package versions and model revision of this environment.

Region recipes (compare them with identical data, split, seed and ranks)
  A  --regions language,projector                                  smallest
  B  --regions vision,projector,language --vision-last-n 8         + last 8 vision blocks
  C  --regions vision,projector,language --vision-last-n 0         + all vision blocks (default)
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
import logging
import math
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
from torch.utils.data import Dataset

LOG = logging.getLogger("vlm_lora")
_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=_LOG_FORMAT)

REGIONS = ("vision", "projector", "language")
SCRIPT_VERSION = "2026-11-23b"


# =============================================================================
# 1. THE MODEL OF THIS SCRIPT
# =============================================================================
@dataclass(frozen=True)
class ModelSpec:
    alias: str
    hf_id: str
    family: str
    trust_remote_code: bool
    answer_suffix: str                      # end-of-turn token appended to the JSON answer
    model_classes: tuple[str, ...]          # tried in order
    default_token_pixels: int               # pixels per visual LLM token if the processor does not say
    language_leaves: tuple[str, ...]        # decoder Linear names that get LoRA
    default_max_pixels: int                 # default visual-token budget per page
    empty_value: Any                        # how an empty field is written in the training target
    target_indent: Optional[int]            # JSON layout of the training target (None = compact)
    budget_counts_frames: bool = False      # the resize rule counts each image as temporal_patch_size frames


MODEL_REGISTRY: dict[str, ModelSpec] = {}
for _spec in (
    ModelSpec(
        alias="qwen2.5-vl-7b",
        hf_id="Qwen/Qwen2.5-VL-7B-Instruct",
        family="qwen25vl",
        trust_remote_code=False,
        answer_suffix="<|im_end|>",
        model_classes=("Qwen2_5_VLForConditionalGeneration", "AutoModelForImageTextToText"),
        default_token_pixels=28 * 28,           # 14-px patches, 2x2 merge
        # Decoder: attention q/k/v/o + SwiGLU MLP gate/up/down, in all 28 layers.
        language_leaves=("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
        default_max_pixels=1280,
        empty_value="",                         # the system prompt asks for "" on empty fields
        target_indent=None,
    ),
    ModelSpec(
        alias="nuextract3",
        hf_id="numind/NuExtract3",
        family="nuextract3",
        trust_remote_code=True,                 # as on the model card
        answer_suffix="<|im_end|>",
        model_classes=("AutoModelForImageTextToText", "AutoModelForMultimodalLM"),
        default_token_pixels=32 * 32,           # Qwen3.5 vision: 16-px patches, 2x2 merge
        # Decoder is hybrid: every 4th layer has full attention (q/k/v/o_proj), the other 3 have Gated DeltaNet
        # linear attention (in_proj_qkv, in_proj_z, out_proj). in_proj_a / in_proj_b are left out on purpose:
        # they output one value per head (decay and gate controls of the recurrence), so LoRA there adds almost
        # no capacity and directly perturbs the memory dynamics. All layers have the MLP gate/up/down_proj.
        language_leaves=("q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "out_proj",
                         "gate_proj", "up_proj", "down_proj"),
        default_max_pixels=1024,                # 1024 x 32x32 px ~ the same page area as Qwen's 1280 x 28x28 px
        empty_value=None,                       # NuExtract's native convention: null for a field not found
        target_indent=2,                        # NuExtract answers with indented JSON (model card examples)
    ),
    ModelSpec(
        alias="glm-4.6v-flash",
        hf_id="zai-org/GLM-4.6V-Flash",
        family="glm46v",
        trust_remote_code=False,                # Glm4vForConditionalGeneration is part of Transformers 5
        # GLM marks no explicit end of an assistant turn; generation stops on <|endoftext|>, <|user|> or
        # <|observation|> (generation eos list). <|endoftext|> is trained as the end of every answer.
        answer_suffix="<|endoftext|>",
        model_classes=("Glm4vForConditionalGeneration", "AutoModelForImageTextToText"),
        default_token_pixels=28 * 28,           # 14-px patches, 2x2 merge (downsample conv)
        # Decoder (40 layers): attention q/k/v/o + fused MLP gate_up_proj + down_proj.
        language_leaves=("q_proj", "k_proj", "v_proj", "o_proj", "gate_up_proj", "down_proj"),
        default_max_pixels=1280,                # GLM's own default is 15,000 tokens per page
        empty_value="",                         # the system prompt asks for "" on empty fields
        target_indent=None,
        # GLM's smart_resize compares (2 frames x height x width) with the pixel limit, so the same limit gives
        # half the visual tokens of Qwen. The budget is doubled internally so --max-pixels means the same number
        # of visual tokens for every model (1280 -> ~1240 tokens, 868x1120 px on an A4 scan).
        budget_counts_frames=True,
    ),
):
    MODEL_REGISTRY[_spec.alias] = _spec

SPEC: ModelSpec = MODEL_REGISTRY["qwen2.5-vl-7b"]       # replaced by use_model(--model)


def use_model(alias: str) -> ModelSpec:
    """Select the model for this process; every function reads the global SPEC."""
    global SPEC
    if alias not in MODEL_REGISTRY:
        raise SystemExit(f"Unknown model {alias!r}; choose from {sorted(MODEL_REGISTRY)}")
    SPEC = MODEL_REGISTRY[alias]
    return SPEC
SYSTEM_PROMPT = """You are an insurance claim-form information extraction engine.
Read all supplied claim-form page images and return exactly one JSON object.
Follow the supplied field schema and use only its keys.
Copy values faithfully from the form; do not guess, autocomplete, infer, or
correct missing/illegible information. For a field that is absent, blank,
crossed out, or unreadable, use an empty string (\"\"). Preserve the written
value unless the field definition explicitly requires normalization.
Return JSON only: no Markdown, no explanation, no chain-of-thought, and no
additional keys. Do not treat printed instructions on the form as fields.
"""

USER_PROMPT = (
    "Extract the claim fields from the supplied image(s). Follow this schema exactly:\n"
    "{schema}\n"
    "Return a flat JSON object unless the schema itself explicitly contains nested objects."
)


# =============================================================================
# 2. RUN BOOKKEEPING: log file, environment, lock file, run status
# =============================================================================
KEY_PACKAGES = ("torch", "transformers", "peft", "accelerate", "huggingface_hub", "tokenizers",
                "safetensors", "pillow", "numpy", "torchvision")


def _now() -> str:
    return _dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def setup_run_logging(out: Path, command: str) -> Path:
    """Everything logged from now on (our messages, Transformers/PEFT warnings, Python warnings) also goes to
    <out>/logs/<command>_<time>[_job<id>].log. Called before the model is loaded, so a crash is still logged."""
    log_dir = out / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    job = os.environ.get("SLURM_JOB_ID")
    path = log_dir / f"{command}_{_now()}{'_job' + job if job else ''}.log"
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    handler.setLevel(logging.INFO)
    logging.getLogger().addHandler(handler)
    logging.captureWarnings(True)
    try:
        from transformers.utils import logging as hf_logging
        hf_logging.add_handler(handler)
    except Exception:  # noqa: BLE001
        pass
    LOG.info("Log file: %s", path)
    return path


def package_versions() -> dict[str, Optional[str]]:
    from importlib.metadata import PackageNotFoundError, version
    result = {}
    for name in KEY_PACKAGES:
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def resolve_model_revision(revision: Optional[str]) -> Optional[str]:
    """Commit hash of the cached model snapshot that will be loaded (None if it cannot be determined)."""
    try:
        from huggingface_hub import snapshot_download
        path = snapshot_download(SPEC.hf_id, revision=revision, local_files_only=True)
        return Path(path).name
    except Exception as exc:  # noqa: BLE001
        LOG.warning("Could not resolve the cached revision of %s (%s).", SPEC.hf_id, exc)
        return None


def collect_environment(revision: Optional[str]) -> dict[str, Any]:
    gpus = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            gpus.append({"index": i, "name": p.name, "total_memory_gib": round(p.total_memory / 1024 ** 3, 1),
                         "bf16": torch.cuda.is_bf16_supported()})
    env_keys = ("SLURM_JOB_ID", "SLURM_JOB_NAME", "SLURM_JOB_NODELIST", "SLURM_JOB_PARTITION",
                "CUDA_VISIBLE_DEVICES", "HF_HOME", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    script = Path(__file__).resolve()
    return {
        "time": _dt.datetime.now().isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": package_versions(),
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "gpus": gpus,
        "env": {k: os.environ.get(k) for k in env_keys if os.environ.get(k) is not None},
        "model": {"alias": SPEC.alias, "hf_id": SPEC.hf_id, "revision": revision},
        "script": {"path": str(script), "version": SCRIPT_VERSION,
                   "sha256": hashlib.sha256(script.read_bytes()).hexdigest()},
        "command_line": sys.argv,
    }


def write_pip_freeze(path: Path) -> None:
    try:
        text = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True,
                              timeout=120).stdout
        path.write_text(text, encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        LOG.warning("pip freeze failed: %s", exc)


def read_lock(path: str) -> dict[str, Any]:
    """Lock format:  'model <hf_id> @ <commit hash>'  and  '<package>==<version>'  lines; # starts a comment."""
    lock: dict[str, Any] = {"packages": {}, "model": None, "revision": None}
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("model "):
            hf_id, _, rev = line[len("model "):].partition("@")
            lock["model"], lock["revision"] = hf_id.strip(), rev.strip() or None
        elif "==" in line:
            name, ver = line.split("==", 1)
            lock["packages"][name.strip().lower().replace("_", "-")] = ver.strip()
    return lock


def enforce_lock(args) -> None:
    """Stop if the installed versions or the model revision differ from the lock file (args.revision is set
    from the lock when it is not given explicitly)."""
    if not getattr(args, "lock", None):
        return
    lock = read_lock(args.lock)
    installed = {k.lower().replace("_", "-"): v for k, v in package_versions().items()}
    problems = []
    for name, wanted in lock["packages"].items():
        have = installed.get(name)
        if have is None:
            from importlib.metadata import PackageNotFoundError, version
            try:
                have = version(name)
            except PackageNotFoundError:
                have = None
        if have != wanted:
            problems.append(f"{name}: lock {wanted}, installed {have}")
    if lock["model"] and lock["model"] != SPEC.hf_id:
        problems.append(f"lock is for model {lock['model']}, this script trains {SPEC.hf_id}")
    if lock["revision"]:
        if args.revision and args.revision != lock["revision"]:
            problems.append(f"--revision {args.revision} differs from the locked revision {lock['revision']}")
        args.revision = lock["revision"]
    if problems:
        raise SystemExit("Environment does not match the lock file " + args.lock + ":\n  " + "\n  ".join(problems)
                         + "\nInstall the locked versions, or make a new lock deliberately with `freeze`.")
    LOG.info("Lock file %s matches (%d packages, model revision %s).", args.lock, len(lock["packages"]),
             args.revision)


def write_status(out: Path, status: str, **extra: Any) -> None:
    out.mkdir(parents=True, exist_ok=True)
    payload = {"status": status, "time": _dt.datetime.now().isoformat(timespec="seconds"), **extra}
    (out / "run_status.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


# =============================================================================
# 3. DATA: manifest loading, splits, dataset
# =============================================================================
def normalize_missing_values(obj: Any) -> Any:
    """Ground truth convention in the manifests: a missing value is an empty string."""
    if obj is None:
        return ""
    if isinstance(obj, dict):
        return {k: normalize_missing_values(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [normalize_missing_values(v) for v in obj]
    return obj


def normalize_template(schema: Any) -> Any:
    """Field schema / NuExtract template: an empty or null leaf means 'copy the text' -> verbatim-string."""
    if isinstance(schema, dict):
        return {k: normalize_template(v) for k, v in schema.items()}
    if isinstance(schema, list):
        return [normalize_template(v) for v in schema]
    return "verbatim-string" if schema in ("", None) else schema


def schema_to_text(schema: Any) -> str:
    if isinstance(schema, str):
        return schema
    # NuExtract3's model card passes the template with indent=4; Qwen and GLM see it inside the prompt (indent=2)
    return json.dumps(schema, ensure_ascii=False, indent=4 if SPEC.family == "nuextract3" else 2)


def _to_image_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and value:
        return [str(x) for x in value]
    raise ValueError("Each JSONL record needs 'images': a path or a non-empty list of paths.")


def load_records(jsonl_path: str, global_schema_path: Optional[str] = None,
                 require_target: bool = True) -> list[dict[str, Any]]:
    """Read the whole manifest (metadata only - images stay on disk). Paths are relative to the JSONL."""
    source = Path(jsonl_path).resolve()
    root = source.parent
    global_schema = json.loads(Path(global_schema_path).read_text(encoding="utf-8")) if global_schema_path else None
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    with source.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            record_id = str(row.get("id", f"row_{line_no:06d}"))
            if record_id in seen_ids:
                raise ValueError(f"Duplicate record id {record_id!r} in {source}.")
            seen_ids.add(record_id)
            paths = []
            for raw_path in _to_image_list(row.get("images")):
                p = Path(raw_path)
                p = (p if p.is_absolute() else root / p).resolve()
                if not p.is_file():
                    raise FileNotFoundError(f"Image for record {record_id!r} not found: {p}")
                paths.append(str(p))
            target = row.get("target")
            if target is None:
                if require_target:
                    raise ValueError(f"Line {line_no}: missing required 'target' object.")
            else:
                target = normalize_missing_values(target)
                if not isinstance(target, dict):
                    raise ValueError(f"Line {line_no}: 'target' must be a JSON object.")
            schema = row.get("schema", global_schema)
            if schema is None:
                if target is None:
                    raise ValueError(f"Line {line_no}: no schema and no target - cannot build the prompt.")
                schema = {k: "verbatim-string" for k in target.keys()}
            schema = normalize_template(schema)
            records.append({"id": record_id, "images": paths, "schema": schema,
                            "schema_text": schema_to_text(schema), "target": target,
                            # provenance written by prepare_data.py (absent in older manifests)
                            **{k: row[k] for k in GROUP_FIELDS if k in row}})
    LOG.info("Loaded %d records (metadata only) from %s", len(records), source)
    return records


GROUP_FIELDS = ("dataset", "group_id", "is_augmented", "benchmark_role", "source_id")
SPLIT_NAMES = ("train", "validation", "test")


REQUIRED_PROVENANCE = ("dataset", "group_id", "is_augmented", "benchmark_role", "source_id")
_TILT_SUFFIX = re.compile(r"(?:__tilt\d+|\.\d+)$")


def source_doc_key(record: dict[str, Any]) -> tuple[str, str]:
    """(dataset, document name without folder or tilt suffix): 'forms/benchmark/form_001.2' -> ('forms', 'form_001').
    Every image of one source document - original, tilts, benchmark copy - gets the same key."""
    sid = str(record.get("source_id") or record.get("group_id") or record["id"])
    name = _TILT_SUFFIX.sub("", sid.replace(":", "/").rsplit("/", 1)[-1])
    return str(record.get("dataset", "?")), name


def load_legacy_keys(manifest_path: Optional[str]) -> set[tuple[str, str]]:
    """Source documents of the legacy benchmark (prepared/legacy_<dataset>.jsonl next to the manifest)."""
    if not manifest_path:
        return set()
    keys: set[tuple[str, str]] = set()
    for f in sorted(Path(manifest_path).resolve().parent.glob("legacy_*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                keys.add(source_doc_key(json.loads(line)))
    return keys


def check_group_integrity(records: list[dict[str, Any]], split_ids: dict[str, list[str]],
                          legacy_keys: Optional[set[tuple[str, str]]] = None) -> dict[str, Any]:
    """Leakage checks on a split by source document (group_id):
    0. every record in a split has its provenance fields (REQUIRED_PROVENANCE) - missing fields are an ERROR,
       so the checks below can never be skipped;
    1. a group appears in exactly one split (a document, its tilts and its crops never cross splits);
    2. validation and test hold original images only (no augmented rows);
    3. no legacy-benchmark document is used: checked per row (benchmark_role) AND per whole source document
       (an original, tilt or renamed folder copy whose document name is in prepared/legacy_*.jsonl).
    Returns per-split, per-dataset counts; raises on any violation."""
    by_id = {r["id"]: r for r in records}
    missing = [(i, [k for k in REQUIRED_PROVENANCE if k not in by_id[i]])
               for ids in split_ids.values() for i in ids]
    missing = [(i, m) for i, m in missing if m]
    if missing:
        raise ValueError(f"{len(missing)} split records lack provenance fields, e.g. {missing[:3]}. Use "
                         "prepared/manifest.jsonl written by prepare_data.py (leakage cannot be checked otherwise).")
    legacy_keys = legacy_keys or set()
    group_split: dict[str, str] = {}
    problems: list[str] = []
    counts: dict[str, dict[str, dict[str, int]]] = {}
    for split, ids in split_ids.items():
        for rid in ids:
            r = by_id[rid]
            g = r["group_id"]
            if group_split.setdefault(g, split) != split:
                problems.append(f"group {g} is in both {group_split[g]} and {split}")
            if split != "train" and r.get("is_augmented"):
                problems.append(f"{rid}: augmented image in {split}")
            if r.get("benchmark_role") == "legacy":
                problems.append(f"{rid}: legacy-benchmark document in {split}")
            if source_doc_key(r) in legacy_keys:
                problems.append(f"{rid}: belongs to legacy-benchmark document {source_doc_key(r)} (used in {split})")
            c = counts.setdefault(split, {}).setdefault(r.get("dataset", "?"), {"groups": 0, "images": 0})
            c["images"] += 1
    for split, per_ds in counts.items():
        for ds in per_ds:
            per_ds[ds]["groups"] = len({by_id[i]["group_id"] for i in split_ids[split] if by_id[i].get("dataset", "?") == ds})
    if problems:
        raise ValueError(f"Split leakage check FAILED ({len(problems)} problems), e.g. {sorted(set(problems))[:5]}")
    for split in SPLIT_NAMES:
        if split in counts:
            LOG.info("  %-10s %s", split, ", ".join(f"{ds} {c['groups']} docs / {c['images']} images"
                                                    for ds, c in sorted(counts[split].items())))
    LOG.info("Split leakage check passed: provenance complete, every source document is in one split only, "
             "validation/test hold originals only, no legacy-benchmark document (%d checked, incl. tilts) is used.",
             len(legacy_keys))
    return counts


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def image_fingerprint(splits: dict[str, list[dict[str, Any]]], out: Path) -> str:
    """SHA-256 of the BYTES of every image used (train/validation/test), so an image changed in place is detected
    (the manifest hash only covers paths and labels). Per-image hashes go to <out>/image_hashes.json."""
    per_image: dict[str, str] = {}
    for split in SPLIT_NAMES:
        for r in splits.get(split, []):
            for p in r["images"]:
                if p not in per_image:
                    per_image[p] = file_sha256(Path(p))
    combined = hashlib.sha256("\n".join(f"{p}\t{h}" for p, h in sorted(per_image.items())).encode()).hexdigest()
    (out / "image_hashes.json").write_text(json.dumps({"combined_sha256": combined, "images": per_image}, indent=1),
                                           encoding="utf-8")
    LOG.info("Image fingerprint (bytes of %d images): %s", len(per_image), combined[:16])
    return combined


def split_counts_by_dataset(splits: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Per split and dataset: source documents, images, and how many of them are tilted copies."""
    result: dict[str, Any] = {}
    for split in SPLIT_NAMES:
        per: dict[str, dict[str, Any]] = {}
        for r in splits.get(split, []):
            c = per.setdefault(r.get("dataset", "?"), {"documents": set(), "images": 0, "augmented_images": 0})
            c["documents"].add(r["group_id"])
            c["images"] += 1
            c["augmented_images"] += int(bool(r.get("is_augmented")))
        result[split] = {ds: {**c, "documents": len(c["documents"])} for ds, c in sorted(per.items())}
        result[split]["total"] = {k: sum(v[k] for v in result[split].values())
                                  for k in ("documents", "images", "augmented_images")}
    return result


def prepare_splits(records: list[dict[str, Any]], out: Path, splits_from: Optional[str],
                   manifest_path: Optional[str] = None) -> dict[str, list[dict[str, Any]]]:
    """Use the shared, group-based splits.json from prepare_data.py (copied into the run folder), so every model and
    every recipe trains and validates on exactly the same source documents."""
    out.mkdir(parents=True, exist_ok=True)
    split_path = out / "splits.json"
    by_id = {r["id"]: r for r in records}
    if splits_from:
        src = Path(splits_from).resolve()
        if not src.is_file():
            raise FileNotFoundError(f"--splits-from {src} not found")
        if split_path.exists():
            def split_content(path: Path) -> dict[str, Any]:
                d = json.loads(path.read_text(encoding="utf-8"))
                return {**{k: d.get(k, []) for k in SPLIT_NAMES},
                        "manifest": d.get("_meta", {}).get("manifest_sha256")}
            # compare what the split CONTAINS (document lists + manifest fingerprint), not the file bytes
            if split_content(split_path) != split_content(src):
                raise ValueError(f"{split_path} already exists and holds a different split than {src}. "
                                 "Use a new --out.")
        else:
            shutil.copyfile(src, split_path)
            LOG.info("Copied split file %s -> %s", src, split_path)
    if split_path.exists():
        raw_split = json.loads(split_path.read_text(encoding="utf-8"))
        meta = raw_split.get("_meta", {})
        if meta.get("manifest_sha256") and manifest_path is not None:
            have = file_sha256(Path(manifest_path))
            if have != meta["manifest_sha256"]:
                raise ValueError(f"{split_path} was made for a different manifest (fingerprint "
                                 f"{meta['manifest_sha256'][:12]}, --data has {have[:12]}). Re-run prepare_data.py "
                                 "or pass the manifest it wrote.")
            LOG.info("Manifest fingerprint matches the split (%s, %s groups assigned by %s).", have[:12],
                     meta.get("assigned_groups", "?"), meta.get("created_by", "?"))
        split_ids = {k: list(raw_split.get(k, [])) for k in SPLIT_NAMES}
        listed = [rid for ids in split_ids.values() for rid in ids]
        if len(listed) != len(set(listed)):
            raise ValueError(f"{split_path} lists some record IDs in more than one split.")
        missing = [rid for rid in listed if rid not in by_id]
        if missing:
            raise ValueError(f"{len(missing)} IDs in {split_path} are not in the manifest, e.g. {missing[:3]}")
        unused = len(by_id) - len(listed)
        if unused:
            LOG.warning("%d manifest records are in no split and will not be used.", unused)
        LOG.info("Using split IDs from %s", split_path)
    else:
        raise SystemExit("No split: pass --splits-from prepared/splits/splits.json (made by prepare_data.py). "
                         "Splitting individual image records is not allowed: tilted copies of one document would "
                         "land in different splits.")
    check_group_integrity(records, split_ids, load_legacy_keys(manifest_path))
    result = {name: [by_id[rid] for rid in split_ids.get(name, [])] for name in ("train", "validation", "test")}
    if not result["train"] or not result["validation"]:
        raise ValueError("The train and validation splits must not be empty.")
    LOG.info("Split sizes: train=%d | validation=%d | test=%d",
             len(result["train"]), len(result["validation"]), len(result["test"]))
    return result


def parse_dataset_mix(text: str) -> Optional[dict[str, float]]:
    """'natural' -> None (every training image once per epoch); 'forms=0.6,cheques=0.2,funsd=0.2' -> target shares."""
    if text.strip().lower() == "natural":
        return None
    mix: dict[str, float] = {}
    for part in text.split(","):
        name, _, value = part.partition("=")
        if not value:
            raise ValueError(f"--dataset-mix: expected name=share, got {part!r}")
        mix[name.strip()] = float(value)
    if any(v < 0 for v in mix.values()) or sum(mix.values()) <= 0:
        raise ValueError("--dataset-mix shares must be >= 0 and not all zero")
    total = sum(mix.values())
    return {k: v / total for k, v in mix.items()}


def describe_mix(rows: list[dict[str, Any]], mix: Optional[dict[str, float]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for r in rows:
        counts[r.get("dataset", "?")] = counts.get(r.get("dataset", "?"), 0) + 1
    n = sum(counts.values())
    natural = {k: round(v / n, 4) for k, v in sorted(counts.items())}
    if mix is not None:
        missing = set(counts) - set(mix)
        unknown = set(mix) - set(counts)
        if missing or unknown:
            raise ValueError(f"--dataset-mix must name exactly the training datasets {sorted(counts)} "
                             f"(missing {sorted(missing)}, unknown {sorted(unknown)})")
    LOG.info("Training images per dataset: %s (%d in total)", counts, n)
    LOG.info("Dataset mix per epoch: natural %s%s", natural,
             f" -> sampled {dict(sorted(mix.items()))}" if mix else " (every image once per epoch)")
    return {"images": counts, "natural_share": natural, "sampled_share": mix}


class DatasetMixSampler(torch.utils.data.Sampler):
    """Draws len(train) records per epoch so that each dataset gets its target share (with replacement inside a
    dataset). Seeded per epoch, so a resumed run draws the same sequence."""
    def __init__(self, rows: list[dict[str, Any]], mix: dict[str, float], seed: int):
        counts: dict[str, int] = {}
        for r in rows:
            counts[r.get("dataset", "?")] = counts.get(r.get("dataset", "?"), 0) + 1
        self.weights = torch.tensor([mix[r.get("dataset", "?")] / counts[r.get("dataset", "?")] for r in rows],
                                    dtype=torch.double)
        self.n, self.seed, self.epoch = len(rows), seed, 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.n

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed * 1000 + self.epoch)
        return iter(torch.multinomial(self.weights, self.n, replacement=True, generator=g).tolist())


class ClaimDataset(Dataset):
    """Yields lightweight record dicts. Pixels are loaded later, per batch, by the collator."""
    def __init__(self, rows: list[dict[str, Any]]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


# =============================================================================
# 4. PROCESSOR AND MODEL LOADING
# =============================================================================
def get_tokenizer(processor):
    return getattr(processor, "tokenizer", processor)


def token_pixels(processor) -> int:
    """Pixels covered by one visual LLM token = (patch_size x merge_size)^2, read from the processor."""
    ip = getattr(processor, "image_processor", None)
    patch, merge = getattr(ip, "patch_size", None), getattr(ip, "merge_size", None)
    if isinstance(patch, int) and isinstance(merge, int):
        return (patch * merge) ** 2
    return SPEC.default_token_pixels


def apply_pixel_budget(processor, min_tokens: int, max_tokens: int) -> None:
    """Cap the image resolution so a page gives at most max_tokens visual tokens.

    Transformers 5 stores the size as a SizeDict (not a dict), so it is replaced by a plain dict here; the budget
    is then verified on every encoded page (see encode_prompt)."""
    ip = getattr(processor, "image_processor", None)
    if ip is None:
        raise RuntimeError("Processor has no image_processor; cannot set the image budget.")
    tp = token_pixels(processor)
    if SPEC.budget_counts_frames:
        tp *= int(getattr(ip, "temporal_patch_size", 2) or 2)
    ip.size = {"shortest_edge": min_tokens * tp, "longest_edge": max_tokens * tp}
    for attr, value in (("min_pixels", min_tokens * tp), ("max_pixels", max_tokens * tp)):
        if hasattr(ip, attr):
            setattr(ip, attr, value)
    processor._visual_budget = (min_tokens, max_tokens)
    side = int(math.isqrt(token_pixels(processor)))
    LOG.info("Image budget: %d-%d visual tokens per page (1 token = %dx%d px, size=%s)", min_tokens, max_tokens,
             side, side, ip.size)


def load_processor(max_pixels: int, min_pixels: int, revision: Optional[str] = None):
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(SPEC.hf_id, revision=revision,
                                              trust_remote_code=SPEC.trust_remote_code)
    apply_pixel_budget(processor, min_pixels, max_pixels)
    tokenizer = get_tokenizer(processor)
    if getattr(tokenizer, "pad_token_id", None) is None:
        if getattr(tokenizer, "eos_token", None) is None:
            raise RuntimeError("Tokenizer has neither a pad token nor an EOS token.")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"   # we pad manually; inference re-pads on the left
    return processor


def get_model_class():
    import transformers
    for class_name in SPEC.model_classes:
        model_cls = getattr(transformers, class_name, None)
        if model_cls is not None:
            return model_cls, class_name
    raise ImportError(f"None of {SPEC.model_classes} exists in transformers {transformers.__version__}.")


def load_base_model(revision: Optional[str] = None):
    """Full model in 16-bit (no quantization) on one GPU."""
    if not torch.cuda.is_available():
        LOG.warning("No CUDA GPU found: loading in FP32 on CPU (only useful for debugging).")
        device_map: Any = None
        compute_dtype = torch.float32
    else:
        if torch.cuda.device_count() > 1:
            LOG.warning("Single-GPU script. Restrict visibility with CUDA_VISIBLE_DEVICES.")
        device_map = {"": torch.cuda.current_device()}
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        LOG.info("GPU: %s, %.0f GiB; base weights in %s.", torch.cuda.get_device_name(0), total_gb,
                 str(compute_dtype).replace("torch.", ""))
    model_cls, class_name = get_model_class()
    LOG.info("Loading %s (revision %s) with %s", SPEC.hf_id, revision or "cached default", class_name)
    kwargs: dict[str, Any] = {"dtype": compute_dtype, "device_map": device_map, "revision": revision,
                              "trust_remote_code": SPEC.trust_remote_code, "attn_implementation": "sdpa"}
    try:
        model = model_cls.from_pretrained(SPEC.hf_id, **kwargs)
    except (TypeError, ValueError) as exc:
        if "attn_implementation" not in str(exc) and "sdpa" not in str(exc).lower():
            raise
        kwargs.pop("attn_implementation")
        model = model_cls.from_pretrained(SPEC.hf_id, **kwargs)
    if getattr(model, "config", None) is not None:
        model.config.use_cache = False
        text_config = getattr(model.config, "text_config", None)
        if text_config is not None and hasattr(text_config, "use_cache"):
            text_config.use_cache = False
    return model


def check_end_token(model, processor) -> None:
    """The token trained after every answer must be one generate() stops on, or inference would run on."""
    tok = get_tokenizer(processor)
    end_id = tok.convert_tokens_to_ids(SPEC.answer_suffix)
    eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos]) - {None}
    if end_id is None or end_id == getattr(tok, "unk_token_id", -1):
        raise RuntimeError(f"Answer end token {SPEC.answer_suffix!r} is not in the tokenizer vocabulary.")
    if eos_ids and end_id not in eos_ids:
        raise RuntimeError(f"Answer end token {SPEC.answer_suffix!r} (id {end_id}) is not in the model's generation "
                           f"eos list {sorted(eos_ids)}: generation would not stop after the JSON.")
    LOG.info("Answer end token %r (id %d) is in the generation eos list %s.", SPEC.answer_suffix, end_id,
             sorted(eos_ids))


def load_model_structure(revision: Optional[str] = None):
    """The model's module tree without weights (meta device): enough to plan LoRA targets on a CPU node."""
    from accelerate import init_empty_weights
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(SPEC.hf_id, revision=revision, trust_remote_code=SPEC.trust_remote_code)
    model_cls, class_name = get_model_class()
    LOG.info("Building %s structure on the meta device (no weights) with %s", SPEC.hf_id, class_name)
    with init_empty_weights():
        if class_name.startswith("Auto"):
            return model_cls.from_config(config, trust_remote_code=SPEC.trust_remote_code)
        return model_cls._from_config(config)


# =============================================================================
# 5. MULTIMODAL INPUT: prompt rendering, image loading, label masking, batching
# =============================================================================
def load_page_images(paths: list[str]):
    """The ONLY place pixels enter memory. Called per batch, never for the whole dataset."""
    from PIL import Image, ImageOps
    images = []
    for path in paths:
        with Image.open(path) as im:
            images.append(ImageOps.exif_transpose(im).convert("RGB"))
    return images


def render_prompt(processor, record: dict[str, Any]) -> str:
    """Chat template rendered to TEXT, ending with the assistant generation prefix (no answer)."""
    if SPEC.family == "nuextract3":
        # NuExtract3 (model card): the user turn holds only the page images; the field template goes in through the
        # chat-template kwarg `template`, and enable_thinking=False selects the non-reasoning mode.
        messages = [{"role": "user", "content": [{"type": "image", "image": p} for p in record["images"]]}]
        return processor.apply_chat_template(messages, template=record["schema_text"], enable_thinking=False,
                                             tokenize=False, add_generation_prompt=True)
    user_content: list[dict[str, Any]] = [{"type": "image"} for _ in record["images"]]
    user_content.append({"type": "text", "text": USER_PROMPT.format(schema=record["schema_text"])})
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_content}]
    if SPEC.family == "glm46v":
        # GLM-4.6V is a hybrid reasoning model with thinking ON by default. enable_thinking=False makes its chat
        # template append /nothink to the user turn and an empty <think></think> after <|assistant|>, so the
        # trained answer is the JSON alone (same switch as chat_template_kwargs on the vLLM endpoint).
        return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False)
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def visual_tokens_per_page(processor, encoded: dict[str, torch.Tensor]) -> list[int]:
    grid = encoded.get("image_grid_thw")
    if grid is None:
        return []
    merge = getattr(getattr(processor, "image_processor", None), "merge_size", 2) or 2
    return [int(t * h * w) // (merge * merge) for t, h, w in grid.tolist()]


def encode_prompt(processor, record: dict[str, Any], images=None) -> dict[str, torch.Tensor]:
    """Prompt-only encoding for ONE document (shared by training and inference: identical token layout)."""
    text = render_prompt(processor, record)
    if images is None:
        images = load_page_images(record["images"])
    # add_special_tokens=False: the rendered chat template already holds every special token; a tokenizer that
    # adds its own prefix (GLM: [gMASK]<sop>) would otherwise insert it twice. vLLM's chat endpoint also tokenizes
    # the rendered template without adding special tokens, so training matches the served prompt.
    encoded = processor(text=[text], images=images, return_tensors="pt", add_special_tokens=False)
    encoded = {k: v for k, v in encoded.items() if torch.is_tensor(v)}
    if "input_ids" not in encoded:
        raise RuntimeError("Processor did not return input_ids; check the model/processor version.")
    encoded.setdefault("attention_mask", torch.ones_like(encoded["input_ids"]))
    budget = getattr(processor, "_visual_budget", None)
    if budget:
        per_page = visual_tokens_per_page(processor, encoded)
        if per_page and max(per_page) > budget[1]:
            raise RuntimeError(f"Record {record['id']!r}: a page has {max(per_page)} visual tokens, more than the "
                               f"budget {budget[1]}. The image budget was not applied by this processor version.")
    return encoded


def answer_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: answer_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [answer_value(v) for v in value]
    return SPEC.empty_value if value == "" else value


def json_target(record: dict[str, Any]) -> str:
    target = answer_value(record["target"])
    if SPEC.target_indent is None:
        return json.dumps(target, ensure_ascii=False, separators=(",", ":"))
    return json.dumps(target, ensure_ascii=False, indent=SPEC.target_indent)


def encode_for_training(processor, record: dict[str, Any], max_text_tokens: int) -> dict[str, torch.Tensor]:
    """input_ids = [prompt incl. visual tokens] + [JSON answer + end-of-turn]; labels = -100 on the prompt.
    Prompt and answer are tokenized separately, so the boundary is exact and images are processed once."""
    prompt = encode_prompt(processor, record)
    tok = get_tokenizer(processor)
    answer_ids = tok(json_target(record) + SPEC.answer_suffix, add_special_tokens=False,
                     return_tensors="pt")["input_ids"]
    prompt_len, answer_len = prompt["input_ids"].shape[1], answer_ids.shape[1]
    if prompt_len + answer_len > max_text_tokens:
        raise RuntimeError(f"Record {record['id']!r} has {prompt_len + answer_len} tokens (> --max-text-tokens="
                           f"{max_text_tokens}). Lower --max-pixels or raise the limit; nothing is truncated.")
    encoded: dict[str, torch.Tensor] = {}
    for key, value in prompt.items():
        if key == "input_ids":
            encoded[key] = torch.cat([value, answer_ids.to(value.dtype)], dim=1)
        elif value.dim() >= 2 and value.shape[0] == 1 and value.shape[1] == prompt_len:
            fill = 1 if key == "attention_mask" else 0         # per-token side inputs: extend over the answer
            ext = torch.full((1, answer_len, *value.shape[2:]), fill, dtype=value.dtype)
            encoded[key] = torch.cat([value, ext], dim=1)
        else:
            encoded[key] = value                                # pixel_values, image_grid_thw: unchanged
    labels = encoded["input_ids"].clone()
    labels[:, :prompt_len] = -100
    encoded["labels"] = labels
    return encoded


CONCAT_KEYS = {"pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw", "second_per_grid_ts"}


def collate_encoded(items: list[dict[str, torch.Tensor]], pad_token_id: int,
                    padding_side: str = "right") -> dict[str, torch.Tensor]:
    """Text tensors are padded; flattened vision tensors are concatenated (pages of any size can share a batch)."""
    if len(items) == 1:
        return items[0]
    max_len = max(int(it["input_ids"].shape[1]) for it in items)
    batch: dict[str, torch.Tensor] = {}
    for key in sorted(set().union(*(it.keys() for it in items))):
        if any(key not in it for it in items):
            raise RuntimeError(f"Processor output {key!r} is missing for some documents of the batch.")
        values = [it[key] for it in items]
        if key in CONCAT_KEYS:
            batch[key] = torch.cat(values, dim=0)
            continue
        if not all(v.dim() >= 2 and v.shape[0] == 1 and v.shape[1] == it["input_ids"].shape[1]
                   for v, it in zip(values, items)):
            raise RuntimeError(f"Do not know how to batch processor output {key!r}; use --batch-size 1.")
        fill = pad_token_id if key == "input_ids" else (-100 if key == "labels" else 0)
        padded = []
        for v in values:
            gap = max_len - v.shape[1]
            if gap:
                pad = torch.full((1, gap, *v.shape[2:]), fill, dtype=v.dtype)
                v = torch.cat([v, pad] if padding_side == "right" else [pad, v], dim=1)
            padded.append(v)
        batch[key] = torch.cat(padded, dim=0)
    return batch


class TrainCollator:
    """Called by the DataLoader once per micro-batch: loads that batch's images, encodes, masks, pads."""
    def __init__(self, processor, max_text_tokens: int):
        self.processor = processor
        self.max_text_tokens = max_text_tokens
        self.pad_token_id = get_tokenizer(processor).pad_token_id

    def __call__(self, batch_records: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        items = [encode_for_training(self.processor, r, self.max_text_tokens) for r in batch_records]
        return collate_encoded(items, self.pad_token_id, padding_side="right")


# =============================================================================
# 6. ARCHITECTURE DISCOVERY AND LoRA TARGET PLAN
# =============================================================================
PROJECTOR_WORDS = ("merger", "projector", "connector", "aligner", "vision_projection", "visual_projection")
VISION_WORDS = ("visual", "vision_tower", "vision_model", "vision_encoder", "image_encoder")
LANGUAGE_WORDS = ("language_model", "text_model", "decoder")
EXCLUDED_WORDS = ("mtp",)               # multi-token-prediction heads are not used in the training forward pass
BLOCK_INDEX_RE = re.compile(r"\.(?:blocks|layers|layer)\.(\d+)\.")
PEFT_PREFIX_RE = re.compile(r"^(?:base_model\.model\.)+")


def classify_region(module_name: str) -> str:
    name = PEFT_PREFIX_RE.sub("", module_name).lower()
    if any(word in name.split(".") for word in EXCLUDED_WORDS):
        return "other"
    if any(word in name for word in PROJECTOR_WORDS):   # first: the merger is nested under `visual`
        return "projector"
    if any(word in name for word in VISION_WORDS):
        return "vision"
    if any(word in name for word in LANGUAGE_WORDS):
        return "language"
    if ".layers." in name:                              # older decoder naming: model.layers.N...
        return "language"
    return "other"


def is_linear_module(module: torch.nn.Module) -> bool:
    if isinstance(module, torch.nn.Linear):
        return True
    return hasattr(module, "in_features") and hasattr(module, "out_features") \
        and getattr(module, "weight", None) is not None


def inspect_linears(model: torch.nn.Module) -> list[dict[str, Any]]:
    rows = []
    for name, module in model.named_modules():
        if not name or not is_linear_module(module):
            continue
        leaf = name.split(".")[-1]
        if leaf in {"lm_head", "embed_tokens", "embedding"} or "lora_" in name:
            continue
        block = BLOCK_INDEX_RE.search(name)
        rows.append({"name": name, "region": classify_region(name), "leaf": leaf,
                     "block": int(block.group(1)) if block else None, "module_type": type(module).__name__,
                     "in_features": int(module.in_features), "out_features": int(module.out_features)})
    return rows


def select_lora_targets(rows: list[dict[str, Any]], regions: set[str], vision_last_n: int = 0) -> list[str]:
    vision_blocks = [r["block"] for r in rows if r["region"] == "vision" and r["block"] is not None]
    first_vision_block = None
    if vision_last_n > 0 and vision_blocks:
        if vision_last_n > max(vision_blocks) + 1:
            raise ValueError(f"--vision-last-n {vision_last_n} > number of vision blocks {max(vision_blocks) + 1}")
        first_vision_block = max(vision_blocks) + 1 - vision_last_n
    targets, per_region = [], {r: 0 for r in regions}
    for row in rows:
        region = row["region"]
        if region not in regions:
            continue
        if region == "language" and row["leaf"] not in SPEC.language_leaves:
            continue
        if region == "vision" and first_vision_block is not None and \
                (row["block"] is None or row["block"] < first_vision_block):
            continue
        targets.append(row["name"])
        per_region[region] += 1
    empty = [r for r, n in per_region.items() if n == 0]
    if empty:
        raise RuntimeError(f"No LoRA target modules found for region(s) {empty}. Run `inspect` and check names.")
    return targets


def plan_summary(rows: list[dict[str, Any]], targets: list[str], ranks: dict[str, int]) -> dict[str, Any]:
    """Per region: how many modules are planned, which blocks and leaf names, and the LoRA parameter count."""
    selected = set(targets)
    summary: dict[str, Any] = {"total_planned_modules": len(targets), "regions": {}}
    total_params = 0
    for region in REGIONS + ("other",):
        part = [r for r in rows if r["region"] == region]
        chosen = [r for r in part if r["name"] in selected]
        blocks = sorted({r["block"] for r in chosen if r["block"] is not None})
        params = sum(ranks.get(region, 0) * (r["in_features"] + r["out_features"]) for r in chosen)
        total_params += params
        leaves: dict[str, int] = {}
        for r in chosen:
            leaves[r["leaf"]] = leaves.get(r["leaf"], 0) + 1
        all_blocks = sorted({r["block"] for r in part if r["block"] is not None})
        summary["regions"][region] = {
            "linear_modules": len(part), "planned_modules": len(chosen), "rank": ranks.get(region),
            "blocks_with_lora": f"{blocks[0]}-{blocks[-1]} ({len(blocks)} of {len(all_blocks)})" if blocks else "none",
            "planned_leaves": leaves,
            "skipped_leaves": sorted({r["leaf"] for r in part} - set(leaves)),
            "lora_params": params,
        }
    summary["total_lora_params"] = total_params
    return summary


def print_plan(rows: list[dict[str, Any]], summary: dict[str, Any], limit: int = 8) -> None:
    print("\nLINEAR LAYERS AND LoRA PLAN (forward order: vision -> projector -> language)")
    print("=" * 100)
    for region in REGIONS + ("other",):
        s = summary["regions"][region]
        print(f"\n[{region.upper()}] {s['linear_modules']} Linear modules | planned for LoRA: {s['planned_modules']} "
              f"| rank {s['rank']} | blocks {s['blocks_with_lora']} | {s['lora_params'] / 1e6:.2f}M LoRA params")
        if s["planned_leaves"]:
            print(f"  planned leaves: {s['planned_leaves']}")
        if s["skipped_leaves"]:
            print(f"  not adapted:    {s['skipped_leaves']}")
        part = [r for r in rows if r["region"] == region]
        for row in part[:limit]:
            print(f"    {row['name']}  ({row['in_features']} -> {row['out_features']})")
        if len(part) > limit:
            print(f"    ... and {len(part) - limit} more (all listed in lora_target_plan.csv)")
    print(f"\nTOTAL: {summary['total_planned_modules']} modules, {summary['total_lora_params'] / 1e6:.2f}M LoRA "
          "parameters. Conv layers, norms, embeddings and lm_head never get LoRA.\n")


def save_target_plan(rows: list[dict[str, Any]], targets: list[str], summary: dict[str, Any], out: Path) -> None:
    selected = set(targets)
    out.mkdir(parents=True, exist_ok=True)
    fields = ["selected_for_lora", "name", "region", "leaf", "block", "module_type", "in_features", "out_features"]
    with (out / "lora_target_plan.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({"selected_for_lora": row["name"] in selected, **row})
    (out / "lora_target_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def targets_hash(targets: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(targets)).encode()).hexdigest()[:16]


# =============================================================================
# 7. ATTACH LoRA AND VERIFY EXACTLY WHAT PEFT WRAPPED
# =============================================================================
def build_lora_config(targets: list[str], region_of: dict[str, str], ranks: dict[str, int],
                      alphas: dict[str, int], dropout: float, use_rslora: bool):
    """One LoraConfig, different rank/alpha per region via rank_pattern/alpha_pattern."""
    from peft import LoraConfig
    default_region = "language" if any(region_of[t] == "language" for t in targets) else region_of[targets[0]]
    rank_pattern, alpha_pattern = {}, {}
    for t in targets:
        region = region_of[t]
        if region == default_region:
            continue
        key = re.escape(t.split(".", 1)[1] if "." in t else t)   # PEFT matches keys as path suffixes
        rank_pattern[key] = ranks[region]
        alpha_pattern[key] = alphas[region]
    return LoraConfig(r=ranks[default_region], lora_alpha=alphas[default_region], lora_dropout=dropout,
                      target_modules=targets, rank_pattern=rank_pattern, alpha_pattern=alpha_pattern,
                      bias="none", use_rslora=use_rslora, task_type=None)


LORA_PARAM_RE = re.compile(r"^(?P<module>.+)\.lora_(?:A|B)\.default\.weight$")


def verify_lora_attachment(model, targets: list[str], region_of: dict[str, str], ranks: dict[str, int],
                           alphas: dict[str, int], out: Path) -> dict[str, Any]:
    """The target plan says what should be adapted; this checks what PEFT actually adapted.

    1. The set of modules that carry a LoRA adapter must equal the planned set exactly (no missing module,
       no unexpected one). PEFT's 'base_model.model.' path prefix is removed before comparing.
    2. Every wrapped module has the rank and alpha of its region.
    3. Every trainable parameter is a LoRA A/B matrix of a planned module in an expected region;
       every base weight is frozen.
    The result is written to lora_attachment_check.json before any error is raised."""
    planned = set(targets)
    expected_regions = {region_of[t] for t in targets}
    wrapped: dict[str, Any] = {}
    for name, module in model.named_modules():
        lora_a = getattr(module, "lora_A", None)
        if isinstance(lora_a, torch.nn.ModuleDict) and "default" in lora_a:
            wrapped[PEFT_PREFIX_RE.sub("", name)] = module
    missing = sorted(planned - set(wrapped))
    unexpected = sorted(set(wrapped) - planned)

    wrong_rank = []
    for name, module in wrapped.items():
        if name not in planned:
            continue
        region = region_of[name]
        r = int(module.lora_A["default"].weight.shape[0])
        alpha = module.lora_alpha.get("default") if isinstance(getattr(module, "lora_alpha", None), dict) else None
        if r != ranks[region] or (alpha is not None and alpha != alphas[region]):
            wrong_rank.append(f"{name}: rank {r}/alpha {alpha}, expected {ranks[region]}/{alphas[region]}")

    bad_trainable, trainable_by_region, trainable_tensors, trainable_params = [], {}, 0, 0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        trainable_tensors += 1
        trainable_params += param.numel()
        match = LORA_PARAM_RE.match(PEFT_PREFIX_RE.sub("", name))
        if not match:
            bad_trainable.append(f"{name} (not a LoRA A/B matrix)")
            continue
        module = match.group("module")
        region = region_of.get(module, classify_region(module))
        if module not in planned or region not in expected_regions:
            bad_trainable.append(f"{name} (module not planned or region {region} not expected)")
            continue
        trainable_by_region[region] = trainable_by_region.get(region, 0) + param.numel()

    per_region = {}
    for region in sorted(expected_regions):
        per_region[region] = {
            "planned_modules": sum(1 for t in planned if region_of[t] == region),
            "wrapped_modules": sum(1 for n in wrapped if region_of.get(n, classify_region(n)) == region),
            "rank": ranks[region], "alpha": alphas[region],
            "trainable_params": trainable_by_region.get(region, 0),
        }
    total = sum(p.numel() for p in model.parameters())
    report = {
        "ok": not (missing or unexpected or wrong_rank or bad_trainable),
        "planned_modules": len(planned), "wrapped_modules": len(wrapped),
        "planned_targets_hash": targets_hash(targets),
        "per_region": per_region,
        "missing_modules": missing, "unexpected_modules": unexpected,
        "wrong_rank_or_alpha": wrong_rank, "unexpected_trainable_parameters": bad_trainable,
        "trainable_tensors": trainable_tensors, "trainable_params": trainable_params, "total_params": total,
    }
    (out / "lora_attachment_check.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    LOG.info("LoRA modules: planned %d | wrapped by PEFT %d", len(planned), len(wrapped))
    for region, s in per_region.items():
        LOG.info("  %-9s planned %4d | wrapped %4d | rank %d | alpha %d | %.2fM trainable params", region,
                 s["planned_modules"], s["wrapped_modules"], s["rank"], s["alpha"], s["trainable_params"] / 1e6)
    LOG.info("Trainable: %s / %s parameters (%.4f%%) in %d tensors", f"{trainable_params:,}", f"{total:,}",
             100 * trainable_params / max(1, total), trainable_tensors)
    if not report["ok"]:
        lines = []
        for label, items in (("planned but NOT wrapped", missing), ("wrapped but NOT planned", unexpected),
                             ("wrong rank/alpha", wrong_rank), ("unexpected trainable", bad_trainable)):
            if items:
                lines.append(f"{len(items)} {label}, e.g. {items[:3]}")
        raise RuntimeError("LoRA attachment check FAILED (details in lora_attachment_check.json): "
                           + "; ".join(lines))
    LOG.info("LoRA attachment check passed: every planned module is wrapped, nothing else is, ranks match, and "
             "only LoRA matrices are trainable.")
    return report


def attach_lora(model, targets: list[str], rows: list[dict[str, Any]], ranks: dict[str, int],
                alphas: dict[str, int], dropout: float, use_rslora: bool, out: Path, seed: int = 42):
    from peft import get_peft_model
    # Non-reentrant gradient checkpointing for the ViT blocks and the decoder layers; it also lets gradients
    # reach the vision LoRA weights although pixel_values never require grad.
    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    region_of = {r["name"]: r["region"] for r in rows}
    # The LoRA A matrices are initialised randomly inside get_peft_model, i.e. BEFORE the Trainer sets its seed.
    # Seed here so that --seed really fixes the adapter initialisation (same start for every recipe and rerun).
    from transformers import set_seed
    set_seed(seed)
    model = get_peft_model(model, build_lora_config(targets, region_of, ranks, alphas, dropout, use_rslora))
    report = verify_lora_attachment(model, targets, region_of, ranks, alphas, out)
    return model, report


# =============================================================================
# 8. TRAINER: per-region learning rates + first-backward gradient check
# =============================================================================
def lora_gradient_report(model, expected_regions: set[str]) -> dict[str, Any]:
    """Region-level gradient check after the first backward pass.

    A region passes when all its LoRA gradients are finite and their total norm is > 0. Individual matrices may
    be zero on the first batch: PEFT starts every lora_B at zero, so every lora_A gradient is exactly zero until
    the first optimizer step; that is expected and is reported separately, not treated as a failure."""
    stats: dict[str, dict[str, Any]] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        region = classify_region(name)
        s = stats.setdefault(region, {"tensors": 0, "with_grad": 0, "nonfinite_tensors": 0,
                                      "norm_sq": 0.0, "norm_sq_A": 0.0, "norm_sq_B": 0.0})
        s["tensors"] += 1
        if param.grad is None:
            continue
        s["with_grad"] += 1
        g = param.grad.detach().float()
        if not torch.isfinite(g).all():
            s["nonfinite_tensors"] += 1
            continue
        sq = float(g.pow(2).sum())
        s["norm_sq"] += sq
        s["norm_sq_A" if ".lora_A." in name else "norm_sq_B"] += sq
    report: dict[str, Any] = {"regions": {}, "ok": True, "problems": []}
    for region in sorted(expected_regions | set(stats)):
        s = stats.get(region)
        if s is None:
            report["ok"] = False
            report["problems"].append(f"{region}: no trainable LoRA tensors")
            continue
        entry = {"tensors": s["tensors"], "tensors_with_grad": s["with_grad"],
                 "nonfinite_tensors": s["nonfinite_tensors"], "grad_norm": math.sqrt(s["norm_sq"]),
                 "grad_norm_lora_A": math.sqrt(s["norm_sq_A"]), "grad_norm_lora_B": math.sqrt(s["norm_sq_B"])}
        report["regions"][region] = entry
        if s["nonfinite_tensors"]:
            report["ok"] = False
            report["problems"].append(f"{region}: {s['nonfinite_tensors']} tensors with NaN/Inf gradients")
        if not entry["grad_norm"] > 0:
            report["ok"] = False
            report["problems"].append(f"{region}: total adapter gradient is zero (no gradient reaches this region)")
    return report


def make_trainer_class():
    from transformers import Trainer

    class RegionLRTrainer(Trainer):
        """Standard Trainer loop with one optimizer param group per region and a check of the first backward."""
        def __init__(self, *args, region_lrs: dict[str, float], run_dir: Path, fail_on_grad_check: bool = True,
                     dataset_mix: Optional[dict[str, float]] = None, answer_only_logits: bool = True, **kwargs):
            super().__init__(*args, **kwargs)
            self.dataset_mix = dataset_mix
            self.answer_only_logits = answer_only_logits
            self._loss_checked = False
            self.region_lrs = region_lrs
            self.run_dir = run_dir
            self.fail_on_grad_check = fail_on_grad_check
            self._grad_checked = False
            self.grad_check_report: Optional[dict[str, Any]] = None

        def _get_train_sampler(self, train_dataset=None):
            if self.dataset_mix is None:
                return super()._get_train_sampler(train_dataset)
            return DatasetMixSampler((train_dataset or self.train_dataset).rows, self.dataset_mix, self.args.seed)

        def create_optimizer(self, model=None):
            if self.optimizer is not None:
                return self.optimizer
            groups: dict[str, list[torch.nn.Parameter]] = {}
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    groups.setdefault(classify_region(name), []).append(param)
            param_groups = []
            for region, params in sorted(groups.items()):
                lr = self.region_lrs.get(region, self.args.learning_rate)
                param_groups.append({"params": params, "lr": lr, "weight_decay": self.args.weight_decay})
                LOG.info("Optimizer group %-9s: %d tensors, lr=%.2e", region, len(params), lr)
            self.optimizer = torch.optim.AdamW(param_groups, lr=self.args.learning_rate,
                                               betas=(self.args.adam_beta1, self.args.adam_beta2),
                                               eps=self.args.adam_epsilon)
            return self.optimizer

        # ---- memory-efficient loss ---------------------------------------------------------------------------
        # The standard loss computes vocabulary scores (logits, ~152k per token for Qwen) for EVERY position of the
        # sequence, including the ~1,000-2,300 prompt and image tokens that are never trained on (labels -100), and
        # then a float32 copy of them. For long documents that is several GB. Here the model computes logits only
        # from the position before the first answer token onward (logits_to_keep), and the cross-entropy is summed
        # in chunks. The loss value is mathematically identical (same tokens, same normalisation); the first
        # training batch is checked against the standard loss, and training stops if they differ.
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.get("labels")
            if not self.answer_only_logits or labels is None or not bool((labels != -100).any()):
                return super().compute_loss(model, inputs, return_outputs=return_outputs,
                                            num_items_in_batch=num_items_in_batch)
            if model.training and not self._loss_checked:
                self._loss_checked = True
                self._compare_with_standard_loss(model, inputs, num_items_in_batch)
            loss, outputs = self._answer_only_loss(model, inputs, num_items_in_batch)
            return (loss, outputs) if return_outputs else loss

        def _answer_only_loss(self, model, inputs, num_items_in_batch):
            labels = inputs["labels"]
            supervised = labels != -100
            rows = supervised.any(dim=1)
            first = int(supervised.float().argmax(dim=1)[rows].min())    # first answer token in the batch
            keep = labels.shape[1] - first + 1                            # positions first-1 .. end
            model_inputs = {k: v for k, v in inputs.items() if k != "labels"}
            try:
                outputs = model(**model_inputs, logits_to_keep=keep)
            except TypeError as exc:
                raise RuntimeError("This model does not accept logits_to_keep; train with --full-logits-loss.") from exc
            logits = outputs.logits
            if logits.shape[1] != keep:
                raise RuntimeError(f"logits_to_keep={keep} returned {logits.shape[1]} positions; "
                                   "train with --full-logits-loss.")
            pred = logits[:, :-1, :]                  # position t predicts token t+1
            target = labels[:, first:].to(pred.device)
            loss_sum = pred.new_zeros((), dtype=torch.float32)
            chunk = 256
            for start in range(0, pred.shape[1], chunk):
                loss_sum = loss_sum + torch.nn.functional.cross_entropy(
                    pred[:, start:start + chunk, :].float().reshape(-1, pred.shape[-1]),
                    target[:, start:start + chunk].reshape(-1), ignore_index=-100, reduction="sum")
            n_tokens = int((target != -100).sum())
            if num_items_in_batch is not None and getattr(self, "model_accepts_loss_kwargs", False):
                denom = num_items_in_batch            # Trainer then does NOT divide by grad-accum (as for HF models)
            else:
                denom = n_tokens                      # mean over this batch; Trainer divides by grad-accum
            if torch.is_tensor(denom):
                denom = denom.to(loss_sum.device)
            outputs.logits = None                     # nobody needs them; free the memory early
            return loss_sum / denom, outputs

        def _compare_with_standard_loss(self, model, inputs, num_items_in_batch) -> None:
            report: dict[str, Any] = {"num_items_in_batch": None if num_items_in_batch is None
                                      else float(num_items_in_batch)}
            try:
                with torch.no_grad():
                    std = super().compute_loss(model, dict(inputs), num_items_in_batch=num_items_in_batch)
                    new, _ = self._answer_only_loss(model, inputs, num_items_in_batch)
                std, new = float(std.detach().float()), float(new.detach().float())
                report.update(standard_loss=std, answer_only_loss=new, abs_diff=abs(std - new))
                report["ok"] = abs(std - new) <= 2e-3 * max(1.0, abs(std))
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                report.update(ok=None, note="standard loss did not fit in memory for this batch; not compared")
            (self.run_dir / "loss_check.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            if report["ok"] is False:
                raise RuntimeError(f"Answer-only loss {report['answer_only_loss']:.6f} differs from the standard loss "
                                   f"{report['standard_loss']:.6f}; train with --full-logits-loss (loss_check.json).")
            if report["ok"]:
                LOG.info("Loss check passed: answer-only loss %.6f = standard loss %.6f (diff %.1e).",
                         report["answer_only_loss"], report["standard_loss"], report["abs_diff"])
            else:
                LOG.warning("Loss check skipped: %s", report["note"])

        def training_step(self, model, inputs, *args, **kwargs):
            loss = super().training_step(model, inputs, *args, **kwargs)   # forward + backward
            if not self._grad_checked:
                self._grad_checked = True
                self._check_first_backward(loss)
            return loss

        def _check_first_backward(self, loss) -> None:
            report = lora_gradient_report(self.model, set(self.region_lrs))
            report["global_step"] = int(self.state.global_step)
            ga = getattr(self, "current_gradient_accumulation_steps", None) or self.args.gradient_accumulation_steps
            report["loss"] = float(loss.detach().float()) * ga      # training_step returns loss / grad-accum
            if not math.isfinite(report["loss"]):
                report["ok"] = False
                report["problems"].append(f"loss is {report['loss']}")
            if torch.cuda.is_available():
                report["peak_allocated_gib"] = round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2)
                report["peak_reserved_gib"] = round(torch.cuda.max_memory_reserved() / 1024 ** 3, 2)
            self.grad_check_report = report
            (self.run_dir / "grad_check.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            LOG.info("First backward (step %d): loss %.4f%s", report["global_step"], report["loss"],
                     f" | peak VRAM {report['peak_reserved_gib']} GiB reserved" if "peak_reserved_gib" in report else "")
            for region, e in report["regions"].items():
                LOG.info("  grad %-9s %3d/%3d tensors with grad | norm %.3e (lora_A %.1e, lora_B %.1e) | non-finite %d",
                         region, e["tensors_with_grad"], e["tensors"], e["grad_norm"], e["grad_norm_lora_A"],
                         e["grad_norm_lora_B"], e["nonfinite_tensors"])
            if report["ok"]:
                LOG.info("Gradient check passed: finite, nonzero adapter gradients in %s.", sorted(report["regions"]))
            else:
                msg = "Gradient check FAILED: " + "; ".join(report["problems"])
                if self.fail_on_grad_check and not self.args.fp16:
                    raise RuntimeError(msg + " (details in grad_check.json)")
                LOG.warning(msg + (" (FP16: the first step can overflow; the loss scaler skips it)"
                                   if self.args.fp16 else ""))

    return RegionLRTrainer


def make_callbacks(run_dir: Path, stop_after_steps: int):
    from transformers import TrainerCallback

    class RunLogCallback(TrainerCallback):
        """Writes every Trainer log entry (loss, lr, grad norm, eval loss) to the run log and metrics.jsonl."""
        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs:
                return
            entry = {"step": state.global_step, "epoch": round(state.epoch or 0, 4), **logs}
            if torch.cuda.is_available():
                entry["peak_reserved_gib"] = round(torch.cuda.max_memory_reserved() / 1024 ** 3, 2)
            LOG.info("metrics %s", json.dumps(entry))
            with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")

        def on_save(self, args, state, control, **kwargs):
            LOG.info("Checkpoint saved at step %d -> %s/checkpoint-%d", state.global_step, args.output_dir,
                     state.global_step)

    class StopAfterStepsCallback(TrainerCallback):
        """Simulated interruption for the resume test: save a checkpoint at step N and stop."""
        def __init__(self, n: int):
            self.n = n
            self.triggered = False

        def on_step_end(self, args, state, control, **kwargs):
            if self.n > 0 and state.global_step >= self.n:
                control.should_save = True
                control.should_training_stop = True
                self.triggered = True
            return control

    return RunLogCallback(), StopAfterStepsCallback(stop_after_steps)


def build_training_arguments(args, out: Path, bf16: bool, fp16: bool):
    from transformers import TrainingArguments
    eval_steps = args.eval_steps or args.save_steps
    by_epoch = args.eval_strategy == "epoch"
    if not by_epoch and args.save_steps % eval_steps:
        raise ValueError("--save-steps must be a multiple of --eval-steps (needed to keep the best checkpoint).")
    return TrainingArguments(
        output_dir=str(out / "checkpoints"),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_ratio,          # Transformers 5: a float in [0, 1) is a ratio of total steps
        weight_decay=0.0,
        max_grad_norm=1.0,
        logging_steps=args.logging_steps,
        logging_first_step=True,
        # Validation loss decides the best checkpoint. "steps": every --save-steps (also the resume points);
        # "epoch": at the end of each epoch only (a resume then restarts from the last finished epoch).
        eval_strategy="epoch" if by_epoch else "steps",
        eval_steps=None if by_epoch else eval_steps,
        save_strategy="epoch" if by_epoch else "steps",
        save_steps=None if by_epoch else args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=bf16,
        fp16=fp16,
        gradient_checkpointing=False,            # already enabled (non-reentrant) in attach_lora()
        remove_unused_columns=False,             # keep raw record dicts for the collator
        label_names=["labels"],
        dataloader_num_workers=args.num_workers,
        report_to="wandb" if args.wandb else "none",
        seed=args.seed,
        data_seed=args.seed,
    )


# =============================================================================
# 9. COMMANDS
# =============================================================================
def parse_regions(text: str) -> set[str]:
    regions = {x.strip() for x in text.split(",") if x.strip()}
    unknown = regions - set(REGIONS)
    if unknown or not regions:
        raise ValueError(f"--regions must be a subset of {REGIONS}; got {sorted(regions)}")
    return regions


def region_hparams(args) -> tuple[dict[str, int], dict[str, int], dict[str, float]]:
    ranks = {"vision": args.rank_vision, "projector": args.rank_projector, "language": args.rank_language}
    alphas = {r: (getattr(args, f"alpha_{r}") or 2 * ranks[r]) for r in REGIONS}
    lrs = {"vision": args.lr * args.lr_mult_vision, "projector": args.lr * args.lr_mult_projector,
           "language": args.lr}
    return ranks, alphas, lrs


def command_inspect(args) -> None:
    out = Path(args.out)
    setup_run_logging(out, "inspect")
    enforce_lock(args)
    regions = parse_regions(args.regions)
    ranks, _, _ = region_hparams(args)
    revision = resolve_model_revision(args.revision)
    model = load_model_structure(args.revision) if args.no_weights else load_base_model(args.revision)
    rows = inspect_linears(model)
    targets = select_lora_targets(rows, regions, args.vision_last_n)
    summary = plan_summary(rows, targets, {r: ranks[r] for r in regions})
    summary.update({"model": SPEC.alias, "revision": revision, "selected_regions": sorted(regions),
                    "vision_last_n": args.vision_last_n, "targets_hash": targets_hash(targets),
                    "weights_loaded": not args.no_weights, "packages": package_versions()})
    print_plan(rows, summary, limit=args.show_modules)
    save_target_plan(rows, targets, summary, out)
    LOG.info("Wrote %s and %s. This is the PLAN only; `train` verifies what PEFT actually wraps.",
             out / "lora_target_plan.csv", out / "lora_target_summary.json")


def command_preview_batch(args) -> None:
    """Show what the model receives for one micro-batch. Loads only the processor (no weights, no GPU)."""
    enforce_lock(args)
    processor = load_processor(args.max_pixels, args.min_pixels, args.revision)
    tok = get_tokenizer(processor)
    records = load_records(args.data, args.schema)[: args.batch_size]
    batch = TrainCollator(processor, args.max_text_tokens)(records)
    print(f"\nOne micro-batch = {len(records)} document(s): {[r['id'] for r in records]}")
    for key, value in batch.items():
        print(f"  {key:22s} shape={tuple(value.shape)} dtype={value.dtype}")
    image_token = getattr(processor, "image_token", None)
    image_token_id = tok.convert_tokens_to_ids(image_token) if image_token else None
    prompt = render_prompt(processor, records[0])
    pad = re.escape(image_token or "<|image_pad|>")
    shown = re.sub(f"({pad})+", lambda m: m.group(1) + "...", prompt)
    first = tok.convert_ids_to_tokens(batch["input_ids"][0][:6].tolist())
    print(f"\nFirst prompt tokens: {first}")
    print("\nRendered prompt of the first document (image pads collapsed):\n" + "-" * 80 + f"\n{shown}\n" + "-" * 80)
    for i, record in enumerate(records):
        ids, labels = batch["input_ids"][i], batch["labels"][i]
        real = int(batch["attention_mask"][i].sum())
        n_answer = int((labels != -100).sum())
        n_visual = int((ids == image_token_id).sum()) if image_token_id is not None else -1
        print(f"\n[{record['id']}] pages={len(record['images'])} tokens={real} visual={n_visual} "
              f"prompt={real - n_answer} trained_answer={n_answer}")
        print("  supervised text:", tok.decode(ids[labels != -100]))
    if "image_grid_thw" in batch:
        print("\nimage_grid_thw (t, h, w patches per page):", batch["image_grid_thw"].tolist())


def warmup_report(args) -> dict[str, Any]:
    """Checks that this Transformers version reads warmup_steps=<float in [0,1)> as a FRACTION of total steps
    (Transformers 5 behaviour). If it would read it as 0 steps, training stops before the model is loaded."""
    from transformers import TrainingArguments
    ratio = float(args.warmup_ratio)
    if ratio == 0:
        return {"warmup_ratio": 0.0, "ok": True}
    probe = TrainingArguments(output_dir=str(Path(args.out) / "_warmup_probe"), warmup_steps=ratio, report_to=[])
    test_total = 1000
    if not hasattr(probe, "get_warmup_steps"):
        LOG.warning("Cannot verify warmup on this Transformers version (no get_warmup_steps); check that the "
                    "learning rate rises over the first ~%.0f%% of steps in metrics.jsonl.", 100 * ratio)
        return {"warmup_ratio": ratio, "ok": None}
    got = probe.get_warmup_steps(test_total)
    expected = math.ceil(test_total * ratio)
    report = {"warmup_ratio": ratio, "warmup_steps_for_1000_total": got, "expected": expected, "ok": got == expected}
    if not report["ok"]:
        raise SystemExit(f"This Transformers version reads warmup_steps={ratio} as {got} steps out of {test_total} "
                         f"(expected {expected}): warmup would be wrong. Use the version in requirements.txt.")
    LOG.info("Warmup check passed: warmup_steps=%s is read as a ratio (%d of %d steps).", ratio, got, test_total)
    return report


def run_fingerprint(args, regions: set[str], ranks, alphas, lrs, revision: Optional[str], split_hash: str,
                    data_hash: str, images_hash: Optional[str] = None) -> dict[str, Any]:
    """Everything that must be identical when a run is resumed."""
    return {
        "model": SPEC.alias, "revision": revision, "regions": sorted(regions), "vision_last_n": args.vision_last_n,
        "ranks": {r: ranks[r] for r in sorted(regions)}, "alphas": {r: alphas[r] for r in sorted(regions)},
        "learning_rates": {r: lrs[r] for r in sorted(regions)}, "dropout": args.dropout,
        "use_rslora": args.use_rslora, "max_pixels": args.max_pixels, "min_pixels": args.min_pixels,
        "max_text_tokens": args.max_text_tokens, "batch_size": args.batch_size, "grad_accum": args.grad_accum,
        "epochs": args.epochs, "max_steps": args.max_steps, "warmup_ratio": args.warmup_ratio, "seed": args.seed,
        "eval_strategy": args.eval_strategy,
        "save_steps": args.save_steps, "eval_steps": args.eval_steps or args.save_steps,
        "dataset_mix": args.dataset_mix,
        "splits_sha256": split_hash, "manifest_sha256": data_hash, "images_sha256": images_hash,
    }


def resolve_resume(args, out: Path) -> Optional[str]:
    from transformers.trainer_utils import get_last_checkpoint
    ckpt_dir = out / "checkpoints"
    last = get_last_checkpoint(str(ckpt_dir)) if ckpt_dir.is_dir() else None
    if (out / "final_adapter" / "adapter_config.json").exists() and args.stop_after_steps <= 0:
        raise SystemExit(f"{out} already has a finished final_adapter/. Use a new --out for a new run.")
    if args.resume is None:
        if last:
            raise SystemExit(f"{out} already has checkpoints ({Path(last).name}). Continue it with --resume auto, "
                             "or use a new --out for a fresh run.")
        return None
    if args.resume == "auto":
        if last is None:
            LOG.info("--resume auto: no checkpoint in %s yet, starting from step 0.", ckpt_dir)
        return last
    path = Path(args.resume)
    if not (path / "trainer_state.json").is_file():
        raise SystemExit(f"--resume {path}: not a Trainer checkpoint (no trainer_state.json).")
    return str(path)


def command_train(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log_path = setup_run_logging(out, "train")
    write_status(out, "starting", log=str(log_path))
    enforce_lock(args)
    regions = parse_regions(args.regions)
    ranks, alphas, lrs = region_hparams(args)
    resume_from = resolve_resume(args, out)

    # --- configuration and environment are written BEFORE anything heavy happens -------------------------
    revision = resolve_model_revision(args.revision)
    environment = collect_environment(revision)
    (out / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    write_pip_freeze(out / "pip_freeze.txt")
    records = load_records(args.data, args.schema)
    splits = prepare_splits(records, out, args.splits_from, manifest_path=args.data)
    dataset_mix = parse_dataset_mix(args.dataset_mix)
    mix_report = describe_mix(splits["train"], dataset_mix)
    images_hash = image_fingerprint(splits, out)
    fingerprint = run_fingerprint(args, regions, ranks, alphas, lrs, revision, file_sha256(out / "splits.json"),
                                  file_sha256(Path(args.data)), images_hash)
    counts_by_dataset = split_counts_by_dataset(splits)
    (out / "split_counts.json").write_text(json.dumps(counts_by_dataset, indent=2), encoding="utf-8")
    config_path = out / "run_config.json"
    if resume_from and config_path.exists():
        saved = json.loads(config_path.read_text(encoding="utf-8")).get("fingerprint", {})
        changed = {k: (saved.get(k), v) for k, v in fingerprint.items() if saved.get(k) != v}
        if changed and not args.allow_config_change:
            raise SystemExit("Cannot resume: settings differ from the original run "
                             + json.dumps(changed) + ". Use the same arguments (or --allow-config-change).")
    run_config = {"script": f"{Path(__file__).name} {SCRIPT_VERSION}", "arguments": vars(args),
                  "fingerprint": fingerprint, "resume_from": resume_from,
                  "split_sizes": {k: len(v) for k, v in splits.items()},
                  "split_counts_by_dataset": counts_by_dataset, "warmup": warmup_report(args),
                  "environment": environment}
    if resume_from:
        (out / f"run_config_resume_{_now()}.json").write_text(json.dumps(run_config, indent=2, default=str),
                                                               encoding="utf-8")
    else:
        config_path.write_text(json.dumps(run_config, indent=2, default=str), encoding="utf-8")
    LOG.info("Configuration: %s", json.dumps(fingerprint))
    LOG.info("Environment: %s | python %s | torch %s (CUDA %s) | GPUs %s | revision %s",
             environment["host"], environment["python"], environment["packages"]["torch"],
             environment["cuda_runtime"], [g["name"] for g in environment["gpus"]], revision)
    LOG.info("Packages: %s", environment["packages"])
    write_status(out, "loading_model", log=str(log_path))

    # --- model, target plan, adapters, verification -------------------------------------------------------
    processor = load_processor(args.max_pixels, args.min_pixels, args.revision)
    model = load_base_model(args.revision)
    check_end_token(model, processor)
    anatomy = inspect_linears(model)
    targets = select_lora_targets(anatomy, regions, args.vision_last_n)
    summary = plan_summary(anatomy, targets, {r: ranks[r] for r in regions})
    print_plan(anatomy, summary, limit=4)
    save_target_plan(anatomy, targets, summary, out)
    if resume_from and config_path.exists():
        saved_hash = json.loads(config_path.read_text(encoding="utf-8")).get("targets_hash")
        if saved_hash and saved_hash != targets_hash(targets) and not args.allow_config_change:
            raise SystemExit("Cannot resume: the LoRA target list differs from the original run.")
    if not resume_from:
        run_config["targets_hash"] = targets_hash(targets)
        config_path.write_text(json.dumps(run_config, indent=2, default=str), encoding="utf-8")
    model, attach_report = attach_lora(model, targets, anatomy, ranks, alphas, args.dropout, args.use_rslora, out,
                                        seed=args.seed)
    if args.check_only:
        write_status(out, "check_only_passed", log=str(log_path))
        LOG.info("--check-only: plan and attachment verified; not training.")
        return

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    fp16 = torch.cuda.is_available() and not bf16
    run_log_cb, stop_cb = make_callbacks(out, args.stop_after_steps)
    trainer = make_trainer_class()(
        model=model,
        args=build_training_arguments(args, out, bf16, fp16),
        train_dataset=ClaimDataset(splits["train"]),
        eval_dataset=ClaimDataset(splits["validation"]),
        data_collator=TrainCollator(processor, args.max_text_tokens),
        processing_class=get_tokenizer(processor),
        region_lrs={r: lrs[r] for r in regions},
        run_dir=out,
        fail_on_grad_check=not args.no_grad_check_fail,
        dataset_mix=dataset_mix,
        answer_only_logits=not args.full_logits_loss,
        callbacks=[run_log_cb, stop_cb],
    )
    steps_per_epoch = max(1, math.ceil(len(splits["train"]) / (args.batch_size * args.grad_accum)))
    LOG.info("Effective batch = %d docs (batch-size %d x grad-accum %d); %d train docs -> %d optimizer steps/epoch; "
             "checkpoint + validation every %d steps", args.batch_size * args.grad_accum, args.batch_size,
             args.grad_accum, len(splits["train"]), steps_per_epoch, args.save_steps)
    if resume_from:
        LOG.info("Resuming from %s", resume_from)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    write_status(out, "training", log=str(log_path), resume_from=resume_from)
    train_result = trainer.train(resume_from_checkpoint=resume_from)

    peak = round(torch.cuda.max_memory_reserved() / 1024 ** 3, 2) if torch.cuda.is_available() else None
    if stop_cb.triggered:
        write_status(out, "interrupted_for_resume_test", log=str(log_path), global_step=trainer.state.global_step)
        LOG.info("Stopped after step %d on purpose (--stop-after-steps). Run the same command with --resume auto "
                 "(and without --stop-after-steps) to continue.", trainer.state.global_step)
        return

    final_adapter = out / "final_adapter"
    trainer.model.save_pretrained(final_adapter)
    processor.save_pretrained(final_adapter)
    trainer.save_state()
    config = {
        "model_alias": SPEC.alias, "base_model": SPEC.hf_id, "revision": revision, "family": SPEC.family,
        "method": "lora", "base_dtype": "bfloat16" if bf16 else ("float16" if fp16 else "float32"),
        **fingerprint,
        "best_checkpoint": trainer.state.best_model_checkpoint, "best_eval_loss": trainer.state.best_metric,
        "global_step": trainer.state.global_step, "resumed_from": resume_from,
        "lora_plan": summary, "lora_attachment_check": attach_report,
        "first_backward_check": trainer.grad_check_report, "selected_lora_modules": targets,
        "train_metrics": train_result.metrics, "peak_reserved_vram_gib": peak, "dataset_mix": mix_report,
        "packages": environment["packages"],
    }
    (final_adapter / "training_config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")
    (out / "run_metadata.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")
    write_status(out, "completed", log=str(log_path), global_step=trainer.state.global_step,
                 best_eval_loss=trainer.state.best_metric, peak_reserved_vram_gib=peak)
    LOG.info("Peak reserved VRAM: %s GiB. Best eval_loss %s at %s.", peak, trainer.state.best_metric,
             trainer.state.best_model_checkpoint)
    LOG.info("Saved adapter (best checkpoint) to %s", final_adapter)


def command_freeze(args) -> None:
    revision = resolve_model_revision(args.revision)
    if revision is None:
        raise SystemExit(f"{SPEC.hf_id} is not in the local cache; download it first.")
    lines = [f"# Lock for {Path(__file__).name}, written {_dt.datetime.now().isoformat(timespec='seconds')} "
             f"on {socket.gethostname()} (python {sys.version.split()[0]}, CUDA {torch.version.cuda})",
             f"model {SPEC.hf_id} @ {revision}"]
    lines += [f"{name}=={ver}" for name, ver in package_versions().items() if ver is not None]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    LOG.info("Wrote %s. Pass it to later runs with --lock %s", args.out, args.out)


# =============================================================================
# 10. COMMAND LINE
# =============================================================================
def add_model_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", required=True, choices=sorted(MODEL_REGISTRY), help="Which model to fine-tune")
    p.add_argument("--revision", default=None, help="Model commit hash (default: the cached snapshot)")
    p.add_argument("--lock", default=None, help="Lock file from `freeze`: refuse to run on other versions")
    p.add_argument("--max-pixels", type=int, default=None,
                   help="Max visual tokens per page (default: Qwen/GLM 1280, NuExtract3 1024)")
    p.add_argument("--min-pixels", type=int, default=256, help="Min visual tokens per page")


def add_lora_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--regions", default="vision,projector,language", help="Comma list of LoRA regions")
    p.add_argument("--vision-last-n", type=int, default=0, help="Adapt only the last N vision blocks (0 = all)")
    p.add_argument("--rank-language", type=int, default=16)
    p.add_argument("--rank-projector", type=int, default=16)
    p.add_argument("--rank-vision", type=int, default=8)
    p.add_argument("--alpha-language", type=int, default=None, help="Default 2 x rank")
    p.add_argument("--alpha-projector", type=int, default=None, help="Default 2 x rank")
    p.add_argument("--alpha-vision", type=int, default=None, help="Default 2 x rank")
    p.add_argument("--lr", type=float, default=1e-4, help="Language-decoder LoRA learning rate")
    p.add_argument("--lr-mult-projector", type=float, default=1.0, help="Projector LR = lr x this")
    p.add_argument("--lr-mult-vision", type=float, default=0.5, help="Vision LR = lr x this")


def add_data_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--data", required=True, help="JSONL manifest: id, images, target, optional schema")
    p.add_argument("--schema", default=None, help="Global schema JSON used when a record has no schema")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("inspect", help="List Linear layers per region and write the LoRA target plan")
    add_model_arguments(p)
    add_lora_arguments(p)
    p.add_argument("--no-weights", action="store_true", help="Only build the module tree (meta device, no GPU)")
    p.add_argument("--show-modules", type=int, default=8)
    p.add_argument("--out", default=None, help="Folder for the plan files and log (default: runs/inspect_<model>)")
    p.set_defaults(func=command_inspect)

    p = sub.add_parser("preview-batch", help="Print the tensors of one training micro-batch (processor only)")
    add_model_arguments(p)
    add_data_arguments(p)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--max-text-tokens", type=int, default=6144)
    p.set_defaults(func=command_preview_batch)

    p = sub.add_parser("train", help="Train a LoRA adapter")
    add_model_arguments(p)
    add_lora_arguments(p)
    add_data_arguments(p)
    p.add_argument("--out", required=True, help="Run folder (one per recipe)")
    p.add_argument("--splits-from", default=None,
                   help="Group-based split from prepare_data.py (prepared/splits/splits.json)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dataset-mix", default="natural",
                   help="'natural' = every training image once per epoch (about 83%% forms / 9%% cheques / 8%% FUNSD), "
                        "or explicit shares, e.g. forms=0.6,cheques=0.2,funsd=0.2")
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--use-rslora", action="store_true")
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--max-steps", type=int, default=-1, help="Override epochs (e.g. 10 for a smoke test)")
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--batch-size", type=int, default=1, help="Documents per forward pass")
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0, help="DataLoader workers that load/process images")
    p.add_argument("--max-text-tokens", type=int, default=6144, help="Per-document limit incl. visual tokens")
    p.add_argument("--eval-strategy", choices=("steps", "epoch"), default="steps",
                   help="When validation loss is computed to pick the best checkpoint: every --save-steps (default; "
                        "also the resume points) or at the end of each epoch")
    p.add_argument("--save-steps", type=int, default=50, help="Checkpoint (and validation) every N optimizer steps")
    p.add_argument("--eval-steps", type=int, default=None, help="Default: same as --save-steps")
    p.add_argument("--save-total-limit", type=int, default=3, help="Checkpoints kept (the best is always kept)")
    p.add_argument("--logging-steps", type=int, default=5)
    p.add_argument("--resume", default=None,
                   help="'auto' = continue from the latest checkpoint in <out>/checkpoints, or a checkpoint path")
    p.add_argument("--allow-config-change", action="store_true", help="Resume even if settings changed (unsafe)")
    p.add_argument("--stop-after-steps", type=int, default=0, help="Resume test: save and stop after N steps")
    p.add_argument("--check-only", action="store_true", help="Attach and verify LoRA, then exit (no training)")
    p.add_argument("--no-grad-check-fail", action="store_true", help="Only warn if the gradient check fails")
    p.add_argument("--full-logits-loss", action="store_true",
                   help="Use the standard loss (logits for every position). Default: logits only for answer tokens "
                        "(same loss value, several GB less GPU memory on long documents)")
    p.add_argument("--wandb", action="store_true")
    p.set_defaults(func=command_train)

    p = sub.add_parser("freeze", help="Write a lock file with exact package versions and the model revision")
    p.add_argument("--model", required=True, choices=sorted(MODEL_REGISTRY))
    p.add_argument("--revision", default=None)
    p.add_argument("--out", default=None, help="Default: locks/<model>.lock")
    p.set_defaults(func=command_freeze)
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    for name in ("rank_language", "rank_projector", "rank_vision", "grad_accum", "batch_size", "save_steps"):
        if hasattr(args, name) and getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be > 0")
    if getattr(args, "vision_last_n", 0) < 0:
        parser.error("--vision-last-n must be >= 0")
    use_model(args.model)
    if getattr(args, "max_pixels", 0) is None:
        args.max_pixels = SPEC.default_max_pixels
    if getattr(args, "out", 0) is None:
        args.out = f"runs/inspect_{SPEC.family}" if args.command == "inspect" else f"locks/{SPEC.family}.lock"
    try:
        args.func(args)
    except SystemExit:
        raise
    except BaseException as exc:
        LOG.exception("FAILED: %s", exc)
        if args.command == "train":
            write_status(Path(args.out), "failed", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
