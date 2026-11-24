#!/bin/bash
# One Slurm job file for all three models.  Choose MODEL, STAGE and (for training) RECIPE:
#   MODEL  = qwen25vl | nuextract3 | glm46v
#   RECIPE = A (language+projector) | B (+ last VLAST vision blocks, default 8) | C (+ all vision blocks)
#   e.g. RECIPE=B,VLAST=10 -> runs/<model>_B10
#
#   sbatch --export=ALL,MODEL=qwen25vl,STAGE=inspect           slurm/job.sbatch   # LoRA plans for A, B, C
#   sbatch --export=ALL,MODEL=qwen25vl,STAGE=check,RECIPE=C    slurm/job.sbatch   # attach + verify LoRA only
#   sbatch --export=ALL,MODEL=qwen25vl,STAGE=smoke,RECIPE=C    slurm/job.sbatch   # 10 steps incl. interrupt + resume
#   sbatch --export=ALL,MODEL=qwen25vl,STAGE=dryrun,RECIPE=C   slurm/job.sbatch   # 1 epoch on the ~100-image subset + scoring
#   sbatch --export=ALL,MODEL=qwen25vl,STAGE=train,RECIPE=A    slurm/job.sbatch   # full run (resubmit = continue)
#   sbatch --export=ALL,MODEL=qwen25vl,STAGE=validate,RECIPE=A slurm/job.sbatch   # score the validation documents
#   sbatch --export=ALL,MODEL=qwen25vl,STAGE=legacy,RECIPE=A,BASE=1 slurm/job.sbatch  # legacy benchmark (+ base once)
#   sbatch --export=ALL,MODEL=qwen25vl,STAGE=final,RECIPE=A    slurm/job.sbatch   # FINAL TEST: only for the chosen runs,
#                                                                                  # after all decisions are made
# Optional: EXTRA="--max-pixels 1024" (same for every recipe of a comparison), MIX="forms=0.6,cheques=0.2,funsd=0.2",
#           EVAL=epoch (validation loss at the end of each epoch instead of every 50 steps)
#SBATCH --job-name=vlm-lora
#SBATCH --partition=CHANGE_ME
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm_%x_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/VLM_Finetuning}"
source .venv/bin/activate
export HF_HOME="${HF_HOME:-$HOME/hf_cache}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
mkdir -p logs runs

MODEL="${MODEL:?set MODEL=qwen25vl|nuextract3|glm46v}"; STAGE="${STAGE:-smoke}"; RECIPE="${RECIPE:-C}"
case "$MODEL" in
  qwen25vl)   ALIAS=qwen2.5-vl-7b ;;
  nuextract3) ALIAS=nuextract3 ;;
  glm46v)     ALIAS=glm-4.6v-flash ;;
  *) echo "MODEL must be qwen25vl, nuextract3 or glm46v"; exit 2 ;;
esac
case "$RECIPE" in
  A) REG="--regions language,projector" ;;
  B) REG="--regions vision,projector,language --vision-last-n ${VLAST:-8}" ;;   # VLAST = last N vision blocks
  C) REG="--regions vision,projector,language --vision-last-n 0" ;;
  *) echo "RECIPE must be A, B or C"; exit 2 ;;
esac
DATA=prepared/manifest.jsonl
SPLITS=prepared/splits/splits.json
LOCK=""; [ -f "locks/$MODEL.lock" ] && LOCK="--lock locks/$MODEL.lock"
EXTRA="${EXTRA:-}"; MIX="${MIX:-natural}"
RUN=runs/${MODEL}_${RECIPE}
[ "$RECIPE" = "B" ] && [ "${VLAST:-8}" != "8" ] && RUN=runs/${MODEL}_B${VLAST}   # e.g. runs/qwen25vl_B10
TRAIN="python scripts/train.py"; INFER="python scripts/infer.py"
echo "[$(date)] MODEL=$MODEL ($ALIAS) STAGE=$STAGE RECIPE=$RECIPE MIX=$MIX on $(hostname) | lock: ${LOCK:-none}"
nvidia-smi --query-gpu=name,memory.total --format=csv || true
[ -f "$DATA" ] || { echo "No $DATA: run  python scripts/prepare_data.py  first"; exit 2; }

case "$STAGE" in
inspect)
  for R in A B C; do
    case $R in A) RR="--regions language,projector";; B) RR="--regions vision,projector,language --vision-last-n 8";;
               C) RR="--regions vision,projector,language --vision-last-n 0";; esac
    $TRAIN inspect --model $ALIAS --no-weights $RR $LOCK --out runs/inspect_${MODEL}_$R
  done
  python scripts/compare_runs.py --plans runs/inspect_${MODEL}_A runs/inspect_${MODEL}_B runs/inspect_${MODEL}_C ;;
check)
  $TRAIN train --model $ALIAS --data $DATA --splits-from prepared/smoke/splits.json $REG $LOCK $EXTRA \
         --out runs/${MODEL}_check_$RECIPE --check-only ;;
smoke)
  OUT=runs/${MODEL}_smoke_$RECIPE; rm -rf "$OUT"
  ARGS="--model $ALIAS --data $DATA --splits-from prepared/smoke/splits.json $REG $LOCK $EXTRA --dataset-mix $MIX
        --out $OUT --max-steps 10 --save-steps 5 --logging-steps 1"
  $TRAIN train $ARGS --stop-after-steps 5          # part 1: stops on purpose after the step-5 checkpoint
  $TRAIN train $ARGS --resume auto                 # part 2: must continue from checkpoint-5 to step 10
  python - "$OUT" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1]); st = json.loads((out / "run_status.json").read_text())
g = json.loads((out / "grad_check.json").read_text()); a = json.loads((out / "lora_attachment_check.json").read_text())
ok = st["status"] == "completed" and st["global_step"] == 10 and g["ok"] and a["ok"] \
     and (out / "final_adapter" / "adapter_model.safetensors").exists()
print("SMOKE", "PASSED" if ok else "FAILED", "| status", st, "| grad ok", g["ok"], "| attach ok", a["ok"])
sys.exit(0 if ok else 1)
PY
  $INFER --adapter $OUT/final_adapter --data $DATA --split validation --max-new-tokens 512 --out $OUT/validation ;;
dryrun)
  # small but complete run on the smoke subset (~100 images with: python scripts/prepare_data.py --smoke-train-docs 11):
  # 1 epoch of training + validation + scoring on the subset's test documents (fine-tuned and base model)
  OUT=runs/${MODEL}_dryrun_$RECIPE; rm -rf "$OUT"
  $TRAIN train --model $ALIAS --data $DATA --splits-from prepared/smoke/splits.json $REG $LOCK $EXTRA \
         --dataset-mix $MIX --epochs "${EPOCHS:-1}" --save-steps 5 --out $OUT
  $INFER --adapter $OUT/final_adapter --data $DATA --split validation --out $OUT/validation
  $INFER --adapter $OUT/final_adapter --data $DATA --split test --compare-base --out $OUT/final_test ;;
train)
  $TRAIN train --model $ALIAS --data $DATA --splits-from $SPLITS $REG $LOCK $EXTRA --dataset-mix $MIX \
         --eval-strategy "${EVAL:-steps}" --out $RUN --resume auto ;;
validate)
  $INFER --adapter $RUN/final_adapter --data $DATA --split validation $LOCK --out $RUN/validation ;;
legacy)
  CMP=""; [ "${BASE:-0}" = "1" ] && CMP="--compare-base"
  for D in forms cheques funsd; do
    $INFER --adapter $RUN/final_adapter --data prepared/legacy_$D.jsonl --split all $CMP $LOCK --out $RUN/legacy_$D
  done ;;
final)
  echo "FINAL TEST: use only after model, recipe, prompt and checkpoint are chosen (on validation / legacy results)."
  $INFER --adapter $RUN/final_adapter --data $DATA --split test --compare-base $LOCK --out $RUN/final_test ;;
*) echo "Unknown STAGE=$STAGE"; exit 2 ;;
esac
echo "[$(date)] done"
