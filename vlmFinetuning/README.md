# VLM_Finetuning: LoRA fine-tuning of Qwen2.5-VL-7B, NuExtract3 and GLM-4.6V-Flash

You split the data yourself by pasting it into folders. The scripts read your folders, check that the split is
clean, train the three models with LoRA, and score them on your test set.

## 1. Folder structure

```
VLM_Finetuning/
├── README.md                      this file
├── requirements.txt               Python packages
├── download_models.sh             downloads the 3 models (login node, once)
├── config/expected_counts.json    how many files each folder should hold (checked automatically)
│
├── dataset/                       >>> YOU PASTE THE DATA HERE <<<   (see dataset/SPLIT_RULES.txt)
│   ├── SPLIT_RULES.txt
│   ├── forms/
│   │   ├── train/        images/  gt_json/    454 forms + 2 tilts each = 1,362 files in each folder
│   │   ├── validation/   images/  gt_json/    57 forms, originals only
│   │   ├── test/         images/  gt_json/    57 forms, originals only
│   │   └── benchmark/    images/  gt_json/    25 forms, originals only
│   ├── cheques/
│   │   ├── train/        70 cheques + 1 tilt each = 140
│   │   ├── validation/   9          test/   8          benchmark/   25
│   └── funsd/
│       ├── train/        45 pages + 2 tilts each = 135
│       ├── validation/   6          test/   5          benchmark/   26 crops
│       └── benchmark/source_pages.txt   the 5 pages the 26 crops were cut from
│
├── scripts/
│   ├── prepare_data.py            reads dataset/, checks the rules, writes prepared/
│   ├── verify_split.py            independent leakage check
│   ├── train.py                   TRAINING (all 3 models)
│   ├── infer.py                   EVALUATION (all 3 models)
│   └── compare_runs.py            all results side by side
├── slurm/job.sbatch               one job file for every step on the GPU cluster
│
├── prepared/                      written by prepare_data.py (do not edit)
├── runs/                          one folder per trained model and option (results)
├── locks/                         exact package versions (written by train.py freeze)
├── logs/                          Slurm outpuat
└── docs/                          the full report (PDF)
```

## 2. Pasting the data

Every folder contains a `README.txt` saying what goes in it and how many files.

| | train (images = gt_json) | validation | test | benchmark |
|---|---|---|---|---|
| forms   | 454 docs → 1,362 (original + 2 tilts) | 57 | 57 | 25 |
| cheques | 70 docs → 140 (original + 1 tilt)      | 9  | 8  | 25 |
| funsd   | 45 docs → 135 (original + 2 tilts)     | 6  | 5  | 26 crops |
| total   | 569 docs → 1,637                       | 72 | 70 | 76 |

File names: original `form_001.png`, tilts `form_001.1.jpg` and `form_001.2.jpg`, ground truth with the same name in
`gt_json/` (`form_001.json`, `form_001.1.json`, `form_001.2.json`). A ground truth is a flat JSON object
(`{"Name": "Ravi Kumar", "Date": "12/03/2024", "Remarks": ""}`).

**The 6 rules**

1. A document and its tilts always stay together, in `train/` only.
2. `validation/` and `test/` hold originals only; their tilts are not used anywhere (not in `train/` either).
3. `benchmark/` documents, and the 5 FUNSD pages the benchmark crops came from, are in no other folder.
4. Choose validation and test documents at random, spread over all layouts/types.
5. Every image has a JSON with the same name in the `gt_json/` folder next to it.
6. Once training has started, never move files between folders.

`prepare_data.py` checks these rules and **stops** if one is broken: the same document in two folders, the same
image under two names in different folders, a tilt whose original is in another folder, a missing ground truth, a
FUNSD benchmark source page in train/validation/test. It **flags** wrong counts (cell marked `!expected`).

## 3. What happens with the data

| Set | Used by | What happens |
|---|---|---|
| train | `train.py` | the model learns from these 1,637 images |
| validation | `train.py` | checked during training; the best checkpoint is kept; you compare options A/B/C here |
| test | `infer.py` | the final score, once, after all choices are made |
| benchmark | `infer.py` | reference only (it was already used to compare models before) |

## 4. Steps

All commands run from the `VLM_Finetuning` folder.

**Once: environment and models (login node)**
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/<your CUDA, e.g. cu126>
pip install -r requirements.txt
export HF_HOME=~/hf_cache && bash download_models.sh
```
Set `#SBATCH --partition` in `slurm/job.sbatch` (find it with `sinfo`).

**Once: prepare the data (login node, CPU)**
```bash
python scripts/prepare_data.py      # every cell must say ok
python scripts/verify_split.py      # must say LEAKAGE CHECK PASSED
```
Run both again whenever you change files in `dataset/`.

**Dry run with ~100 images (recommended first)**
```bash
python scripts/prepare_data.py --smoke-train-docs 11                 # subset: 88 train, 6 validation, 3 test images
sbatch --export=ALL,MODEL=qwen25vl,STAGE=dryrun,RECIPE=C slurm/job.sbatch   # 1 epoch + validation + test scoring
```
Results in `runs/qwen25vl_dryrun_C/`. The subset is taken from your real split; the full split is not changed.

**Per model** (`MODEL=qwen25vl`, `nuextract3` or `glm46v`)
```bash
sbatch --export=ALL,MODEL=qwen25vl,STAGE=smoke,RECIPE=C   slurm/job.sbatch    # quick test: must print SMOKE PASSED
python scripts/train.py freeze --model qwen2.5-vl-7b                           # lock the package versions

sbatch --export=ALL,MODEL=qwen25vl,STAGE=train,RECIPE=A   slurm/job.sbatch    # option A
sbatch --export=ALL,MODEL=qwen25vl,STAGE=train,RECIPE=B   slurm/job.sbatch    # option B (VLAST=10 for 10 blocks)
sbatch --export=ALL,MODEL=qwen25vl,STAGE=train,RECIPE=C   slurm/job.sbatch    # option C

sbatch --export=ALL,MODEL=qwen25vl,STAGE=validate,RECIPE=A slurm/job.sbatch   # then B, C
python scripts/compare_runs.py runs/qwen25vl_A runs/qwen25vl_B runs/qwen25vl_C # choose the best option

sbatch --export=ALL,MODEL=qwen25vl,STAGE=final,RECIPE=B   slurm/job.sbatch    # final test, chosen option only
```
If a training job stops, send the same command again: it continues from its last checkpoint.

## 5. The options A, B, C

Each option is a separate training run from the same original model, with the same data and settings:

| Option | What is trained | Qwen2.5-VL-7B | GLM-4.6V-Flash |
|---|---|---|---|
| A | projector + language | 198 layers, 40.7M parameters | 244 layers, 48.6M |
| B | + last 8 vision blocks (VLAST=10: last 10) | 238 layers, 42.1M | 284 layers, 50.3M |
| C | + all vision blocks | 358 layers, 46.3M | 364 layers, 53.6M |

Choose the smallest option that is not clearly worse on validation.

## 6. Where the results are

```
runs/qwen25vl_A/
├── final_adapter/                       the trained model (best checkpoint)
├── run_metadata.json                    best validation loss, steps, GPU memory
├── validation/eval_summary_validation.json   scores on validation
├── legacy_forms|cheques|funsd/eval_summary_all.json   scores on the benchmark
├── final_test/eval_summary_test.json    final scores (chosen option only)
├── final_test/predictions_test.jsonl    every prediction next to its ground truth
├── metrics.jsonl, logs/, checkpoints/   training history, log, saved progress
runs/comparison.csv                      all runs side by side (compare_runs.py)
```

Scores: Strict Form Accuracy, Precision, Recall, F1, edit similarity, ANLS, overall and per dataset with the number
of documents. Cheques (8) and FUNSD (5) test results are marked PRELIMINARY (fewer than 30 documents).

## 7. Models

| `--model` | Hugging Face ID | `MODEL=` | Endpoint (base model) |
|---|---|---|---|
| qwen2.5-vl-7b | Qwen/Qwen2.5-VL-7B-Instruct | qwen25vl | http://10.68.0.39:9005/v1 |
| nuextract3 | numind/NuExtract3 | nuextract3 | http://10.68.0.39:9003/v1 |
| glm-4.6v-flash | zai-org/GLM-4.6V-Flash (thinking off) | glm46v | http://10.68.0.39:9007/v1 |
