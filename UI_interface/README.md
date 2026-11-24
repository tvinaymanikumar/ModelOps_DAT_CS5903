# Document Extraction

A simple Streamlit app that extracts information from document images with
vision-language models. It works on any document; there are no datasets to
set up.

**The whole flow is one page:**

1. **Add images** (JPG, PNG, WEBP, BMP, TIFF, GIF or PDF).
2. **Choose a schema option:**
   - **No schema**: the model returns every field it finds.
   - **With schema**: type the fields you need, e.g. `name, age, policy_number, claim_amount`.
     The result has exactly those fields.
3. **Extract.** Images are processed one at a time; results appear below with
   JSON, TXT, PDF and ZIP downloads.

Everything runs offline with the **Mock Model**, so you can test on Windows
before connecting a GPU server or API.

---

## 1. Setup (Windows)

You need Python 3.10 or newer (python.org; tick *Add python.exe to PATH*).

**Easiest: double-click two files**

1. Extract the ZIP into a **new, empty folder** (do not copy it over an older
   version: leftover or missing files cause `ModuleNotFoundError`).
2. Double-click **`setup_windows.bat`**. It creates `.venv`, installs the
   packages, creates `.env` from `.env.example` (already pointing at the team
   endpoints) and runs the setup check, including the endpoints.
3. Double-click **`run_app.bat`**. The app opens in your browser.

**Or in a terminal (PowerShell / VS Code)**

```powershell
cd DAT_Document_Extraction
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
python check_setup.py --network
streamlit run app.py
```

If PowerShell blocks `Activate.ps1`, run once:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` (or use the `.bat` files,
which do not need it). In VS Code: `Ctrl+Shift+P` → *Python: Select
Interpreter* → `.venv`.

**If something goes wrong, run `python check_setup.py`.** It checks the Python
version, every project file, the packages, the project imports and `.env`,
and prints `OK` or `FAIL` with the fix for each. Add `--network` to also test
the Qwen, NuExtract3 and GLM endpoints.

Run the automated tests with `pytest`.

## 2. Using the app

The sidebar shows only the **Model** and **Save results as**. Everything else
(handwriting transcription, model parameters, rate limits, image
preprocessing, mock behaviour) is under **Advanced settings**.

**With schema, step by step**

- Type field names separated by commas or new lines. That is all you need.
- *Field types and hints (optional)*: every field is text unless you change it.
  Set number, whole number, date or yes/no, untick *Required* for optional
  fields, and add a hint such as "amount in figures, not words".
- *Load or save a field list*: reuse a saved list (four examples are in
  `presets/`), import a JSON template (`{"name": "", "age": 0}`) or JSON Schema,
  or save the current fields for next time.
- *Preview the output format* shows the JSON the model must return.
- *Edit the prompt (optional)* for full control.

Extract stays disabled until there is at least one field; the reason is shown
under the button. Stop finishes the current image and cancels the rest;
Retry failed reruns only the images that failed.

**Other tabs**: *Earlier runs* (every run is kept), *Evaluation* (score a run
against ground truth), *Settings* (test model connections, delete runs).

**Handwriting**: turn on *Handwriting transcription* under Advanced settings
to get plain-text transcriptions instead of fields.

### Field types

| Type | Accepted value | Example coercion |
|---|---|---|
| string | any text | `5.9` → `"5.9"` |
| integer | whole number or `""` | `"Rs. 2,223,000/-"` → `2223000` |
| number | number or `""` | `"$1,200.50"` → `1200.5` |
| date | text, checked to be a readable date | `"19/10/2018"` |
| boolean | `true` / `false` or `""` | `"yes"` → `true` |

Every field may be `""` so the model can leave an illegible value empty
instead of guessing. The field list generates both the **template** sent to the
model (`{"name": "", ...}`) and the formal **JSON Schema** used for
validation and, where the model supports it, constrained decoding.

### Prompts

Default prompts are the team's prompts, in `prompts/` (`schema_free.txt`,
`schema_based.txt`, and `pipeline_*.txt` for the reconciliation step). With a
schema, the schema block is added automatically; put `{template}` in a prompt
to place the template yourself. Edits in the UI apply to the current session
and are saved with each job.

### Review rules

Rules flag results for review and never change values. They are generated from
your task: required fields that came back empty, dates that cannot be read, too
many empty fields, too few fields (schema-free), empty transcriptions. Flagged
items get status **Needs review**.

## 3. Models

| Model | No schema | With schema | Transcription | How it works |
|---|---|---|---|---|
| **Qwen2.5-VL-7B** | yes | yes | yes | the team's *Qwen only* approach: one Qwen call, then parse, flatten and enforce the schema |
| **NuExtract3** | yes | yes | yes | image + a NuExtract template built from your fields; no schema = NuExtract first generates a template for the page |
| **GLM-4.6V-Flash** | yes | yes | yes | same flow as Qwen (prompt + schema block); thinking is switched off, `<think>` / box markers are removed |
| Mock Model | yes | yes | yes | offline fake, for testing |
| Gemini (optional) | yes | yes | yes | external API, hidden unless `ALLOW_EXTERNAL_MODELS=true` |

### Connecting to the team endpoints

`.env.example` already holds the team's servers; `copy .env.example .env` is enough:

```
QWEN_BASE_URL=http://10.68.0.39:9005/v1        QWEN_MODEL_NAME=qwen2.5-vl-7b
NUEXTRACT_BASE_URL=http://10.68.0.39:9003/v1   NUEXTRACT_MODEL_NAME=nuextract3
GLM_BASE_URL=http://10.68.0.39:9007/v1         GLM_MODEL_NAME=glm-4.6v-flash
```

Your computer must reach 10.68.0.39 (same network or VPN). *Settings → Test
connections* calls `/v1/models` on each endpoint and checks the
model id is served. Requests: prompt first for Qwen, image first for GLM, image
only (template in `chat_template_kwargs`) for NuExtract3; JPEG with longest side 1600 px,
temperature 0.0, top-p 1.0, 3000 max tokens (all adjustable under
*Advanced settings*).

### How the schema is applied

With schema, the prompt gets the scripts' **REQUIRED PREDEFINED SCHEMA** block
built from your fields. Fields typed *whole number* must come back as JSON
integers (the cheque `amount` rule: `"1,80,000/-"` → `180000`); *number*
fields as JSON numbers; everything else as strings. After the reply,
`enforce_schema` keeps exactly your keys (matching ignores case, spaces,
hyphens and underscores), fills missing keys with `""` and converts the
values. If your fields include `age` and `sex`, the NMRDS normalization
(`"12/f"` → `"12 years"`, `"female"`) runs unless you turn it off under
*Advanced settings → Post-processing*.

Without a schema, Qwen returns the fields it finds; nested objects are
flattened to dotted keys (`bank.name`) and every value is a string, as in
`flatten_json`.

### Testing offline

Choose **Mock Model**: it needs no server, GPU or key.

## 4. Architecture

```
app.py                      Streamlit entry point (tabs)
frontend/                   sidebar, upload, task panel (fields), prompt, processing, results, evaluation, settings
config/                     settings (.env), task_config (modes), model_registry, processor_registry
prompts/                    default prompt per mode + shared rules
presets/                    reusable field lists (JSON)
models/                     adapters: endpoint, auth, request format, retries, response extraction
processors/                 generic schema_free / schema_based / transcription + custom/
validation/                 JSON parsing, field list → template / schema, coercion, validation, rules
backend/                    router, job state, background worker, rate limiter, json_validator, outputs
evaluation/                 kv_metrics (KV P / R / F1, NED, strict form accuracy), CER / WER
tests/                      pytest suite (no network or GPU)
```

Per image:

```
image → preprocessing → router → processor → adapter(s) → raw response
      → parse → template + schema check → review rules → saved immediately
```

**Routing priority:** custom processor chosen in the sidebar →
`task.custom_processor` → the model's own processor (team Qwen-script processor) →
generic processor for the mode.

**Batch processing:** one image per request in a background thread (one per
session, never duplicated by reruns). Stop takes effect after the current
request. Retries on timeouts, 429 and 5xx use exponential backoff and count
toward the rate limit (Gemini defaults to 15 requests per minute).

**Item statuses:** completed, needs review, parse failed (no JSON could be
read; never saved as `{}`), failed, cancelled.

## 5. Outputs

```
outputs/<job_id>/
  json/       result envelope per page (result, status, flags, latency, hashes)
  raw/        untouched model response
  txt/        readable report, or the transcription
  pdf/        ReportLab report with the image, fields and flags
  summary/    job.json (task, fields, schema, prompt, settings), summary.csv, audit.jsonl
  evaluation/ metrics.json, per_document_metrics.csv, dataset_summary.csv, fields.csv
```

## 6. Evaluation (optional)

Upload ground truth named like the processed pages (`claim_01.json`,
`form_p2.json`, `page.txt` for transcription) or as one ZIP.

Extraction jobs are scored with the team's evaluator (`evaluation/kv_metrics.py`,
ported from `Metric_new.py`). There are no scoring settings.

1. Keys are aligned by exact / alias match, then (no schema only) by lexical
   similarity of at least 0.80, never pairing opposites such as admission/discharge.
2. A pair is a TP only when the value matches after normalization (case, spaces,
   currency, thousands separators, dates, age units, M/F). A wrong value counts
   as one FP and one FN; extra keys are FP, missing keys FN.
3. Reported: combined KV precision, recall and F1 (micro-averaged), whole-form
   NED similarity (macro-averaged) and strict form accuracy.

Transcription jobs: CER, WER, line exact match.

Jobs with the same task name and mode can be compared side by side, for
example the same claim fields run on Qwen, NuExtract3 and GLM.

## 7. Privacy and security

- Self-hosted models and the Mock Model are the defaults.
- External models are hidden unless `ALLOW_EXTERNAL_MODELS=true`, and then need
  a per-session confirmation before Start.
- Credentials only in `.env` (git-ignored). Logs redact keys and never contain
  image data, extracted values or raw responses; `audit.jsonl` stores hashes only.
- Each job has its own `uploads/<job_id>` and `outputs/<job_id>`; delete them
  from the *Earlier runs* tab (*Delete this run*).

## 8. Extending

**A. New kind of document.** Nothing to code: define its fields in the UI (or
add `presets/<name>.json`) and choose a mode.

**B. Documents needing custom logic.** Add a processor in `processors/custom/`
and register it in `config/processor_registry.py`; it appears in the sidebar.
See `processors/custom/README.md`.

**C. New model**, e.g. a LoRA-fine-tuned Qwen served by vLLM:

```python
# models/qwen_lora_adapter.py
from config.settings import env
from models.openai_compatible import OpenAICompatibleAdapter

class QwenLoRAAdapter(OpenAICompatibleAdapter):
    name = "Qwen2.5-VL-7B-LoRA"
    def __init__(self, **options):
        super().__init__(base_url=env("QWEN_LORA_BASE_URL"),
                         model_name=env("QWEN_LORA_MODEL_NAME", "claims-lora"), **options)
```

Register it in `config/model_registry.py`:
`"Qwen2.5-VL-7B-LoRA": ModelSpec(QwenLoRAAdapter)` (add `modes=[...]` to limit
it). Serve with `vllm serve Qwen/Qwen2.5-VL-7B-Instruct --enable-lora --lora-modules claims-lora=/path/to/adapter`.

**D. New model with an unusual response format.** Normalize it inside its
adapter (`ModelResponse(raw_text=..., parsed=dict_or_None)`), as
`models/glm_adapter.py` does. The UI and batch system are unchanged.

**E. New review rule.** Add a function with `@rule("name")` in
`validation/rules.py` and generate it in `rules_for_task()`.

## 9. Tests

```powershell
pytest
```

Covers field lists → template / schema (and back from presets, templates and
schemas), coercion and validation, JSON parsing, routing, the Mock adapter in
every mode, review rules, the pipeline merge, all metrics, sequential
processing, Stop, retry, duplicate-worker prevention and rate limiting.

## 11. Troubleshooting

| Symptom | Fix |
|---|---|
| Start is disabled | The reason is shown under the buttons (no fields, model not configured, empty prompt). |
| Model shows "not configured" | Fill its variables in `.env`, restart, use *Settings → Test connections*. |
| `ModuleNotFoundError` when starting | Extract the ZIP into a new empty folder, run `setup_windows.bat`, then `python check_setup.py`. |
| Every item is `parse_failed` | Open *Raw response*; tighten the prompt or turn on constrained decoding. |
| Many items need review | Check the flags; relax *Review rules* or mark fields as not required. |
| Non-English text shows boxes in PDFs | Set `PDF_FONT_PATH` to a TTF that covers the script. |
| Large scans are slow | Set *Max long side* in *Image preprocessing* (e.g. 2048). |
