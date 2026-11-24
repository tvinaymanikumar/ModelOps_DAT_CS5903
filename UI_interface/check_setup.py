"""Check that the project is complete and ready to run.

    python check_setup.py            files, Python packages, imports, .env
    python check_setup.py --network  also checks the Qwen, NuExtract3 and GLM endpoints

Every line says OK or FAIL with what to do. Run it from the project folder.
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

REQUIRED_FILES = [
    '.streamlit/config.toml',
    'app.py',
    'backend/__init__.py',
    'backend/batch_processor.py',
    'backend/job.py',
    'backend/json_validator.py',
    'backend/output_manager.py',
    'backend/processing_manager.py',
    'backend/rate_limiter.py',
    'backend/result_manager.py',
    'backend/router.py',
    'config/__init__.py',
    'config/model_registry.py',
    'config/processor_registry.py',
    'config/settings.py',
    'config/task_config.py',
    'evaluation/__init__.py',
    'evaluation/evaluator.py',
    'evaluation/ground_truth.py',
    'evaluation/key_matching.py',
    'evaluation/metrics.py',
    'evaluation/kv_metrics.py',
    'frontend/__init__.py',
    'frontend/evaluation_ui.py',
    'frontend/processing_ui.py',
    'frontend/prompt_ui.py',
    'frontend/results_ui.py',
    'frontend/schema_ui.py',
    'frontend/settings_ui.py',
    'frontend/sidebar.py',
    'frontend/state.py',
    'frontend/upload_ui.py',
    'models/__init__.py',
    'models/base_adapter.py',
    'models/dat_endpoints.py',
    'models/gemini_adapter.py',
    'models/glm_adapter.py',
    'models/mock_adapter.py',
    'models/nuextract_adapter.py',
    'models/openai_compatible.py',
    'models/qwen_adapter.py',
    'processors/__init__.py',
    'processors/base_processor.py',
    'processors/custom/__init__.py',
    'processors/custom/qwen_scripts.py',
    'processors/dat_extraction.py',
    'processors/schema_based.py',
    'processors/schema_free.py',
    'processors/transcription.py',
    'prompts/schema_based.txt',
    'prompts/schema_free.txt',
    'prompts/transcription.txt',
    'requirements.txt',
    'utils/__init__.py',
    'utils/file_utils.py',
    'utils/image_utils.py',
    'utils/logger.py',
    'utils/pdf_report.py',
    'utils/pdf_utils.py',
    'validation/__init__.py',
    'validation/json_parser.py',
    'validation/rules.py',
    'validation/schema_validator.py',
]

PACKAGES = {  # import name -> pip name
    "streamlit": "streamlit", "PIL": "pillow", "pandas": "pandas", "pymupdf": "pymupdf",
    "reportlab": "reportlab", "jsonschema": "jsonschema", "rapidfuzz": "rapidfuzz",
    "requests": "requests", "dotenv": "python-dotenv", "openai": "openai",
}

PROJECT_MODULES = [
    "config.settings", "config.task_config", "config.model_registry", "config.processor_registry",
    "models.dat_endpoints", "models.qwen_adapter", "models.nuextract_adapter", "models.glm_adapter",
    "models.mock_adapter", "evaluation.kv_metrics",
    "processors.dat_extraction", "processors.custom.qwen_scripts",
    "backend.router", "backend.processing_manager", "frontend.sidebar", "frontend.schema_ui",
]

failures = 0


def report(ok: bool, label: str, fix: str = "") -> None:
    global failures
    if ok:
        print(f"  OK    {label}")
    else:
        failures += 1
        print(f"  FAIL  {label}")
        if fix:
            print(f"        -> {fix}")


def main() -> int:
    network = "--network" in sys.argv
    print(f"Project folder: {ROOT}")

    print("\n1. Python")
    report(sys.version_info >= (3, 10), f"Python {sys.version.split()[0]}",
           "Install Python 3.10 or newer from python.org")
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    report(in_venv, "running inside a virtual environment",
           "Run setup_windows.bat, or activate it: .\\.venv\\Scripts\\Activate.ps1")

    print("\n2. Project files")
    missing = [f for f in REQUIRED_FILES if not (ROOT / f).is_file()]
    report(not missing, f"{len(REQUIRED_FILES) - len(missing)} of {len(REQUIRED_FILES)} files present",
           "Files are missing (first ones: " + ", ".join(missing[:5]) + "). Extract the ZIP into a NEW, "
           "empty folder instead of copying over an older version.")
    stale = [p for p in ("processors/custom/paddle_qwen_pipeline.py", "models/paddleocr_client.py",
                         "config/dataset_registry.py", "models/deepseek_adapter.py", "models/mineru_adapter.py",
                         "models/custom_adapter.py", "models/script_adapter.py",
                         "prompts/pipeline_schema_free.txt", "prompts/pipeline_schema_based.txt")
             if (ROOT / p).exists()]
    report(not stale, "no files left over from older versions",
           "Delete: " + ", ".join(stale) + " (or use a fresh folder)")

    print("\n3. Python packages")
    for mod, pip_name in PACKAGES.items():
        try:
            importlib.import_module(mod)
            report(True, pip_name)
        except ImportError:
            report(False, pip_name, "pip install -r requirements.txt")

    print("\n4. Project modules")
    for mod in PROJECT_MODULES:
        try:
            m = importlib.import_module(mod)
            inside = Path(getattr(m, "__file__", "")).resolve().is_relative_to(ROOT)
            report(inside, mod, f"'{mod}' was loaded from {m.__file__}, not from this project. "
                                "Another package with the same name is installed; run from the project folder.")
        except Exception as exc:
            report(False, mod, f"{type(exc).__name__}: {exc}")

    print("\n5. Settings (.env)")
    env_file = ROOT / ".env"
    report(env_file.is_file(), ".env exists", "copy .env.example .env")
    try:
        from config.settings import env
        for key in ("QWEN_BASE_URL", "QWEN_MODEL_NAME", "NUEXTRACT_BASE_URL", "NUEXTRACT_MODEL_NAME",
                    "GLM_BASE_URL", "GLM_MODEL_NAME"):
            report(bool(env(key)), f"{key} = {env(key) or '(empty)'}",
                   f"Set {key} in .env (see .env.example)")
    except Exception as exc:
        report(False, "read settings", str(exc))

    if network:
        print("\n6. Endpoints")
        try:
            from config.model_registry import availability
            for name in ("Qwen2.5-VL-7B", "NuExtract3", "GLM-4.6V-Flash"):
                ok, reason = availability(name)
                report(ok, f"{name}: {reason}", "Check the URL in .env, your network/VPN, and that the server is running")
        except Exception as exc:
            report(False, "endpoint check", f"{type(exc).__name__}: {exc}")
    else:
        print("\n(Skipping endpoint checks. Run: python check_setup.py --network)")

    print()
    if failures:
        print(f"{failures} problem(s) found. Fix the FAIL lines above, then run this again.")
        return 1
    print("Everything is ready. Start the app with: streamlit run app.py  (or run_app.bat)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
