"""Mock Model: deterministic fake outputs for any task.

Lets the whole UI, batch system, validation and evaluation run with no GPU,
API key or network. Outputs are seeded by the image content, so the same image
always gives the same result. For schema-based tasks it fills whatever fields
you defined, using each field's type. Error paths can be exercised with
failure_rate / invalid_json_rate.
"""
from __future__ import annotations

import hashlib
import json
import random
import threading
import time
from typing import Any

from models.base_adapter import AdapterError, BaseAdapter, ModelRequest, ModelResponse, TransientAdapterError

_TEXT_BY_HINT = {
    "name": ["Kumar Ravi Palli", "Anita Sharma", "Debdeep Mukherjee", "J. L. McGinnis"],
    "company": ["INDAH GIFT & HOME DECO", "AEON CO. (M) BHD.", "Sanyu Stationery"],
    "address": ["27, Jalan Dedap 13, Taman Johor Jaya, 81100 Johor Bahru", "House 14, Dum Dum Road, Kolkata"],
    "sex": ["female", "male"], "gender": ["Female", "Male"],
    "email": ["someone@example.com"], "phone": ["(336) 335-7392", "91-9745678912"],
    "policy": ["POL-4471-2290", "MI/22/118803"], "account": ["630801551452", "004501234567"],
    "words": ["Twenty Two lakh Twenty Three thousand", "Seventy Three Lakhs"],
}
_DATES = ["19/10/2018", "22/03/2018", "12/10/98", "25/01/2016"]
_LINES = [
    "A MOVE to stop Mr. Gaitskell from",
    "nominating any more Labour life Peers",
    "is to be made at a meeting of Labour",
    "M Ps tomorrow. Mr. Michael Foot has",
    "put down a resolution on the subject",
]


class MockAdapter(BaseAdapter):
    name = "Mock Model"
    is_external = False
    supports_guided_json = True
    description = "Deterministic fake model for testing the full workflow offline."

    def __init__(self, delay_s: float = 0.6, failure_rate: float = 0.0,
                 invalid_json_rate: float = 0.0, empty_rate: float = 0.15, **options: Any):
        super().__init__(**options)
        self.delay_s = float(delay_s)
        self.failure_rate = float(failure_rate)
        self.invalid_json_rate = float(invalid_json_rate)
        self.empty_rate = float(empty_rate)

    def check_available(self) -> tuple[bool, str]:
        return True, "Always available (offline mock)"

    def _rng(self, request: ModelRequest) -> random.Random:
        digest = hashlib.sha256(request.image.tobytes()[:200_000]).hexdigest()
        return random.Random(int(digest[:12], 16))

    def generate(self, request: ModelRequest, cancel_event: threading.Event | None = None) -> ModelResponse:
        self.check_cancel(cancel_event)
        start = time.perf_counter()
        rng = self._rng(request)
        waited = 0.0
        while waited < self.delay_s:  # small steps keep Stop responsive
            time.sleep(min(0.1, self.delay_s - waited))
            waited += 0.1
        chaos = random.random()  # failure injection independent of the content seed
        if chaos < self.failure_rate:
            raise TransientAdapterError("Mock transient failure (injected)")
        mode = request.params.get("mode", "schema_free")
        if chaos < self.failure_rate + self.invalid_json_rate and mode != "transcription":
            raw = "Sure! Here is the data: {name: unquoted, broken"
        elif mode == "transcription":
            raw = "\n".join(_LINES[:rng.randint(2, len(_LINES))])
        elif mode == "schema_based":
            raw = json.dumps(self._schema_based(request, rng), ensure_ascii=False, indent=2)
        elif mode == "schema_free":
            raw = self._schema_free(rng)
        else:
            raise AdapterError(f"Mock adapter does not know mode '{mode}'")
        return ModelResponse(raw_text=raw, latency_ms=(time.perf_counter() - start) * 1000,
                             peak_memory_mb=None, metadata={"mock": True})

    def _value(self, key: str, ftype: str, rng: random.Random) -> Any:
        k = key.lower()
        if ftype == "integer" or (ftype == "string" and k in {"age", "count", "quantity", "pages"}):
            return rng.randint(1, 90) if "age" in k else rng.choice([2223000, 7300000, 48500, 3])
        if ftype == "number" or any(w in k for w in ("amount", "total", "price", "claim")) and ftype != "string":
            return round(rng.uniform(5, 5000), 2)
        if ftype == "boolean":
            return rng.choice([True, False])
        if ftype == "date" or "date" in k or "dob" in k:
            return rng.choice(_DATES)
        for hint, values in _TEXT_BY_HINT.items():
            if hint in k:
                return rng.choice(values)
        if "total" in k or "amount" in k:
            return f"{rng.uniform(5, 500):.2f}"
        return f"sample {key}"

    def _schema_based(self, request: ModelRequest, rng: random.Random) -> dict[str, Any]:
        fields = request.params.get("fields") or [{"key": k, "type": "string"} for k in (request.params.get("template") or {})]
        out = {f["key"]: self._value(f["key"], f.get("type", "string"), rng) for f in fields}
        for key in list(out):
            if rng.random() < self.empty_rate:  # an illegible field is left empty
                out[key] = ""
        return out

    def _schema_free(self, rng: random.Random) -> str:
        # Raw text deliberately repeats NAME / OFFICE (sender and recipient blocks)
        # and nests a table, to exercise duplicate-key handling and flattening.
        name = rng.choice(_TEXT_BY_HINT["name"])
        return json.dumps({"DATE": rng.choice(_DATES), "TIME": "10:15"}, indent=2)[:-2] + (
            f',\n  "NAME": "{name}",\n  "OFFICE": "Lorillard - New York",\n'
            '  "NAME": "R. H. Orcutt",\n  "OFFICE": "Lorillard Tobacco Company - Houston",\n'
            '  "Educational Qualifications": {"1. Matric": {"Board": "CBSE", "Year": "2013"},'
            ' "2. 10+2": {"Board": "CBSE", "Year": ""}},\n  "Remarks": ""\n}')
