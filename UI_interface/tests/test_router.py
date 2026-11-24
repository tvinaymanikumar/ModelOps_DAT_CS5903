import threading

import pytest
from PIL import Image

from backend.router import RoutingError, resolve_processor_path, route
from models.mock_adapter import MockAdapter
from processors.base_processor import ProcessingContext
from processors.custom.qwen_scripts import QwenOnlyProcessor
from processors.schema_based import SchemaBasedProcessor
from processors.schema_free import SchemaFreeProcessor
from processors.transcription import TranscriptionProcessor
from tests.helpers import make_task, prompt_for


@pytest.mark.parametrize("mode,cls", [("schema_free", SchemaFreeProcessor), ("schema_based", SchemaBasedProcessor),
                                      ("transcription", TranscriptionProcessor)])
def test_generic_routing(mode, cls):
    proc, adapter, path = route(mode, "Mock Model")
    assert isinstance(proc, cls) and isinstance(adapter, MockAdapter) and path == f"generic:{mode}"


@pytest.mark.parametrize("mode", ["schema_free", "schema_based"])
def test_self_hosted_models_use_team_processor(mode):
    for model in ("Qwen2.5-VL-7B", "NuExtract3", "GLM-4.6V-Flash"):
        proc, _, _ = route(mode, model, adapter=MockAdapter(delay_s=0))
        assert type(proc) is QwenOnlyProcessor


def test_qwen_transcription_uses_generic():
    proc, _, path = route("transcription", "Qwen2.5-VL-7B", adapter=MockAdapter(delay_s=0))
    assert isinstance(proc, TranscriptionProcessor) and path == "generic:transcription"


def test_unsupported_combination_and_unknown_processor():
    with pytest.raises(RoutingError):
        route("transcription", "DeepSeek-VL2")
    with pytest.raises(RoutingError):
        resolve_processor_path("schema_based", "Mock Model", processor_choice="nope")


def test_registered_custom_processor_wins(monkeypatch):
    import backend.router as r
    monkeypatch.setitem(r.CUSTOM_PROCESSORS, "Mine", {"path": "processors.schema_free:SchemaFreeProcessor",
                                                      "modes": ["schema_based"]})
    assert resolve_processor_path("schema_based", "Mock Model", "Mine") == "processors.schema_free:SchemaFreeProcessor"
