from config.model_registry import MODEL_REGISTRY, models_for_mode
from config.task_config import MODES, default_prompt


def test_every_mode_has_a_default_prompt_and_models():
    for mode in MODES:
        assert default_prompt(mode).strip()
        assert "Mock Model" in models_for_mode(mode)


def test_only_the_wanted_models_are_registered():
    assert set(MODEL_REGISTRY) == {"Mock Model", "Qwen2.5-VL-7B", "NuExtract3", "GLM-4.6V-Flash", "Gemini"}


def test_external_models_hidden_by_default(monkeypatch):
    import config.model_registry as mr
    monkeypatch.setattr(mr, "ALLOW_EXTERNAL_MODELS", False)
    assert "Gemini" not in mr.models_for_mode("schema_free")
    monkeypatch.setattr(mr, "ALLOW_EXTERNAL_MODELS", True)
    assert "Gemini" in mr.models_for_mode("schema_free")


def test_registry_entries_are_valid():
    for name, spec in MODEL_REGISTRY.items():
        assert set(spec.modes) <= set(MODES), name
