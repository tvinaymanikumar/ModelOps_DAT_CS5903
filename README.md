# ModelOps: End-to-End Document Information Extraction & VLM Fine-Tuning
**Course: Distributed AI Training (CS5903) — Project Repository**

An end-to-end ModelOps system for visual document understanding, information extraction (IE), and vision-language model (VLM) fine-tuning across diverse document domains (bank cheques, official forms, handwritten notes, invoices, and medical records).

---

## 📑 Table of Contents
- [Project Overview](#-project-overview)
- [Repository Structure](#-repository-structure)
- [Prerequisites & Environment Setup](#-prerequisites--environment-setup)
- [Security & API Keys](#-security--api-keys)
- [How to Run: Streamlit Web UI](#-1-how-to-run-the-streamlit-web-ui)
- [How to Run: Base Model Pipelines](#-2-how-to-run-base-model-pipelines)
- [How to Run: VLM Fine-Tuning & Evaluation](#-3-how-to-run-vlm-fine-tuning--evaluation)
- [Evaluation & Metrics](#-evaluation--metrics)
- [Results & Demos](#-results--demos)

---

## 🔍 Project Overview

This repository contains the complete ModelOps lifecycle for document extraction:
1. **Interactive Streamlit UI (`UI_interface`)**: Upload single/batch document images or PDFs, apply predefined or custom JSON schemas, and run extraction through local team endpoints, cloud VLMs (Gemini), or offline mock models.
2. **Multi-Model Evaluation Pipelines (`program_codes`)**: Standardized batch pipelines and evaluation scripts for base models:
   - **Qwen2.5-VL-7B** (PaddleOCR + Qwen pipeline)
   - **DeepSeek-VL2**
   - **MinerU 2.5**
   - **Gemini 3.5 Flash**
   - **NuExtract 3**, **GLM-4.6V-Flash** & **IBM Granite Vision 3.1-2B** (OpenAI-compatible vLLM benchmarks)
3. **Parameter-Efficient VLM Fine-Tuning (`vlmFinetuning`)**: LoRA fine-tuning infrastructure for **Qwen2.5-VL-7B**, **NuExtract3**, and **GLM-4.6V-Flash**, complete with automated dataset verification, Slurm cluster scripts, loss visualization, and evaluation benchmarks.
4. **Benchmark Datasets & Results**: Evaluation on forms6000, Funds Handwritten, IDRBT Cheques, SROIE Invoices, and Nigerian Medical Records.

---

## 📂 Repository Structure

```
ModelOps_DAT_CS5903/
├── UI_interface/                     # Production-ready Streamlit document extraction app
│   ├── app.py                        # Streamlit entry point
│   ├── check_setup.py                # Environment and connectivity diagnostics
│   ├── run_app.bat / setup_windows   # Windows 1-click launch scripts
│   ├── backend/                      # Batch runner, job routers, rate limiters
│   ├── frontend/                     # Modular Streamlit UI views
│   ├── models/                       # Model adapters (Qwen, NuExtract, GLM, Gemini, Mock)
│   ├── presets/                      # Standard document schemas (cheques, invoices, etc.)
│   └── validation/                   # JSON schemas, parsing, and data validation
│
├── program_codes/                    # Model evaluation & extraction pipelines
│   ├── benchmark.py                  # OpenAI-compatible vLLM benchmark runner
│   ├── Deepseek/                     # DeepSeek-VL2 batch scripts & flow pipelines
│   ├── Gemini/                       # Gemini 3.5 Flash batch extraction scripts with .env support
│   ├── MinerU/                       # MinerU 2.5 batch inference & test scripts
│   ├── Padder_Qwen/                  # PaddleOCR + Qwen2.5-VL hybrid extraction pipelines
│   └── Metrics.py / Metric_new.py    # Key-value accuracy, Precision, Recall, F1 metrics
│
├── vlmFinetuning/                    # LoRA fine-tuning module (Qwen2.5-VL, NuExtract3, GLM)
│   ├── download_models.sh            # One-time model caching script
│   ├── scripts/
│   │   ├── prepare_data.py           # Split validation, tilt checking, and manifest generation
│   │   ├── verify_split.py           # Data leakage verification
│   │   ├── train.py                  # LoRA training script with evaluation
│   │   ├── infer.py                  # Test-set batch inference with trained adapters
│   │   └── plot_loss.py              # Loss curve plotting utility
│   ├── slurm/job.sbatch              # Slurm batch execution script for GPU clusters
│   └── prepared/                     # Manifests and split definitions
│
├── Datasets/                         # Ground truth and benchmark document sets
│   ├── forms6000/                    # Government/tax/standard forms
│   ├── Funds_handwritten_dataset/    # Handwritten receipts and notes
│   ├── IDRBT_cheque_dataset/         # Indian banking cheques
│   ├── Invoice_dataset/              # Receipts and commercial invoices
│   └── Nigeria_medical_records/      # Handwritten hospital/clinic forms
│
├── outputs_basemodels/               # Baseline model predictions and metrics summaries
├── vlmFinetuning_demo outputs/       # Interactive HTML comparison reports (Base vs. Fine-tuned)
├── vlmFinetuning_results/            # Training loss plots, validation graphs, and benchmark scores
├── requirements_backup.txt           # Consolidated package snapshot
└── .gitignore                        # Strict rules excluding secrets, venvs, and cache
```

---

## ⚙️ Prerequisites & Environment Setup

### 1. Requirements
- **OS**: Windows, Linux, or macOS
- **Python**: 3.10 or 3.11 recommended
- **Hardware**:
  - *Inference (UI Mock / External APIs)*: Any CPU
  - *Local VLM Inference*: 16 GB+ VRAM GPU (or remote endpoints)
  - *VLM LoRA Fine-Tuning*: 24 GB+ VRAM GPU (A100 / RTX 3090 / RTX 4090)

### 2. Virtual Environment Installation

```bash
# Clone the repository
git clone https://github.com/tvinaymanikumar/ModelOps_DAT_CS5903.git
cd ModelOps_DAT_CS5903

# Create and activate virtual environment
python -m venv .venv

# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1

# On Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements_backup.txt
```

---

## 🔒 Security & API Keys

Sensitive configuration files (`.env`) are excluded from Git to prevent exposing credentials.

### Setting up UI Interface Environment:
1. Navigate to `UI_interface/`:
   ```bash
   cp UI_interface/.env.example UI_interface/.env
   ```
2. Edit `UI_interface/.env` with your team endpoints or external API key:
   ```ini
   ALLOW_EXTERNAL_MODELS=true
   GEMINI_API_KEY=your_actual_gemini_api_key_here
   GEMINI_MODEL_NAME=gemini-3.5-flash
   QWEN_BASE_URL=http://<host>:<port>/v1
   ```

### Setting up Gemini 3.5 Flash Batch Pipeline Environment:
1. Navigate to `program_codes/Gemini/`:
   ```bash
   cp program_codes/Gemini/.env.example program_codes/Gemini/.env
   ```
2. Insert your key into `program_codes/Gemini/.env`:
   ```ini
   GEMINI_API_KEY=your_actual_gemini_api_key_here
   GEMINI_MODEL=gemini-3.5-flash
   ```

---

## 🚀 1. How to Run the Streamlit Web UI

The Streamlit UI provides an interactive web application for single or batch document processing.

### Windows (Quick Launch):
Double-click `UI_interface/run_app.bat` to launch the application.

### From Command Line:
```bash
cd UI_interface

# Optional: Verify environment & endpoints connectivity
python check_setup.py --network

# Launch Streamlit app
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### App Workflow:
1. **Upload Documents**: Drag and drop images (`PNG`, `JPG`, `TIFF`) or `PDF` files.
2. **Select Extraction Mode**:
   - **Schema-Free**: Discovers and extracts all available key-value pairs automatically.
   - **Schema-Based**: Constrains output to predefined schemas (e.g., Bank Cheque, Invoice, Patient Record) or custom user fields.
3. **Select Model**: Choose between **Mock Model** (offline CPU testing), team endpoints (**Qwen2.5-VL**, **NuExtract3**, **GLM-4.6V**), or **Gemini**.
4. **Export**: Download extracted data in JSON, CSV, PDF summary reports, or a consolidated ZIP.

---

## 🔬 2. How to Run Base Model Pipelines

All evaluation pipelines are organized inside `program_codes/`:

### A. Gemini Batch Extraction
```bash
python program_codes/Gemini/batch_cheques.py
python program_codes/Gemini/batch_schema.py
python program_codes/Gemini/noschema_batch.py
```

### B. PaddleOCR + Qwen2.5-VL Pipeline
Combines PaddleOCR text bounding box localization with Qwen2.5-VL semantic key-value extraction:
```bash
python program_codes/Padder_Qwen/pipeline_batch.py
python program_codes/Padder_Qwen/pipeline_batch_cheques.py
python program_codes/Padder_Qwen/pipeline_singlefile.py
```

### C. DeepSeek-VL2 Batch Processing
```bash
python program_codes/Deepseek/1Forms600_batch.py
python program_codes/Deepseek/IDRBT_flow.py
python program_codes/Deepseek/Nigeria_flow.py
```

### D. MinerU 2.5 Extraction
```bash
python program_codes/MinerU/batch_cheques.py
python program_codes/MinerU/batch_schema.py
```

### E. OpenAI-Compatible vLLM Benchmarks (NuExtract 3, GLM-4.6V-Flash & IBM Granite Vision)

Benchmark image-to-JSON extraction models served by an OpenAI-compatible vLLM API:

```bash
# Benchmark NuExtract 3
uv run python benchmark.py \
  --base-url http://localhost:8000/v1 \
  --model nuextract3 \
  --concurrency 4 \
  --output results/nuextract3.json

# Benchmark GLM-4.6V-Flash
uv run python benchmark.py \
  --base-url http://localhost:8000/v1 \
  --model glm-4.6v-flash \
  --concurrency 4 \
  --output results/glm-4.6v-flash.json

# Benchmark IBM Granite Vision 3.1-2B
uv run python benchmark.py \
  --base-url http://localhost:8000/v1 \
  --model ibm-granite/granite-vision-3.1-2b-preview \
  --concurrency 4 \
  --output results/granite-vision.json   
```
*(Run from `program_codes/` or specify `program_codes/benchmark.py`)*

---

## 🧠 3. How to Run VLM Fine-Tuning & Evaluation

The `vlmFinetuning/` directory contains full LoRA fine-tuning workflows for **Qwen2.5-VL-7B**, **NuExtract3**, and **GLM-4.6V-Flash**.

### Step 1: Pre-download Models (Login node / local workstation)
```bash
cd vlmFinetuning
bash download_models.sh
```

### Step 2: Validate Data & Generate Manifests
Checks folder structures, detects potential data leakage, and generates `manifest.jsonl`:
```bash
python scripts/prepare_data.py
python scripts/verify_split.py
```

### Step 3: Train with LoRA
Train any of the supported models (e.g. Qwen2.5-VL-7B):
```bash
python scripts/train.py \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --train-manifest prepared/manifest.jsonl \
    --epochs 3 \
    --batch-size 4 \
    --lora-r 16 \
    --lora-alpha 32 \
    --output-dir runs/qwen2.5_lora_run1
```

*For GPU Cluster Execution (Slurm):*
```bash
sbatch slurm/job.sbatch
```

### Step 4: Run Inference with Fine-Tuned Adapter
```bash
python scripts/infer.py \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --adapter runs/qwen2.5_lora_run1 \
    --input-folder dataset/cheques/test/images \
    --output-dir runs/qwen2.5_test_results
```

### Step 5: Plot Loss & Compare Runs
```bash
python scripts/plot_loss.py --log-file runs/qwen2.5_lora_run1/train.log
python scripts/compare_runs.py --runs-dir runs/
```

---

## 📊 Evaluation & Metrics

Compute exact-match and normalized character/token metrics (Precision, Recall, F1) across model predictions against ground truth:

```bash
# Evaluate predictions against ground truth
python program_codes/Metric_new.py

# Evaluate Qwen-specific pipeline outputs
python program_codes/Metrics_new_qwen.py
```

Metrics computed:
- **Key-Value Match (Strict)**: Exact key and value string match.
- **Key-Value Match (Normalized)**: Case-insensitive, whitespace-trimmed, punctuation-invariant.
- **Character Error Rate (CER)** & **Word Error Rate (WER)**.

---

## 📈 Results & Demos

- **Interactive HTML Demos**: Explore side-by-side base vs. fine-tuned extractions in `vlmFinetuning_demo outputs/` (open any `report.html` in your browser).
- **Evaluation Curves**: Loss graphs and benchmark plots are saved in `vlmFinetuning_results/`.
- **Baseline Predictions**: CSV and JSON performance logs across models are available in `outputs_basemodels/evaluation_metrics/`.
