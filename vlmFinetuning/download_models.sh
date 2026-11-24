#!/bin/bash
# Run ONCE on the login node (it has internet). Downloads all three models into $HF_HOME and checks that they load offline,
# the way the Slurm jobs load them.   Usage: bash download_models.sh   [qwen25vl|nuextract3|glm46v]
set -euo pipefail
source .venv/bin/activate
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
python - "${1:-all}" <<'PY'
import os, sys
from huggingface_hub import snapshot_download
models = {"qwen25vl": "Qwen/Qwen2.5-VL-7B-Instruct", "nuextract3": "numind/NuExtract3",
          "glm46v": "zai-org/GLM-4.6V-Flash"}
pick = sys.argv[1]
for fam, repo in models.items():
    if pick not in ("all", fam):
        continue
    path = snapshot_download(repo)          # ~17 GB (Qwen) / ~10 GB (NuExtract3) / ~21 GB (GLM)
    print(f"{repo}: revision {os.path.basename(path)}  ({path})")
os.environ["HF_HUB_OFFLINE"] = "1"
from transformers import AutoConfig, AutoProcessor
for fam, repo in models.items():
    if pick not in ("all", fam):
        continue
    try:
        AutoConfig.from_pretrained(repo, trust_remote_code=(fam == "nuextract3"))
        AutoProcessor.from_pretrained(repo, trust_remote_code=(fam == "nuextract3"))
        print(f"OK   {repo} loads offline")
    except Exception as e:
        print(f"FAIL {repo}: {type(e).__name__}: {e}  (usually: transformers too old)")
PY
