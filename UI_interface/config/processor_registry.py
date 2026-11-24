"""Optional custom processors the user can pick in the sidebar.

Generic processors cover every mode. Register a custom processor here when a
document type needs special preprocessing, several endpoints, or unusual
parsing; it then appears in the sidebar's "Processing pipeline" list for the
modes it supports. No frontend change is needed.
"""
from __future__ import annotations

CUSTOM_PROCESSORS: dict[str, dict] = {
    # "Cheque with words-vs-figures check": {
    #     "path": "processors.custom.my_cheque:ChequeProcessor",
    #     "modes": ["schema_based"],
    # },
}


def processors_for_mode(mode: str) -> list[str]:
    return [name for name, spec in CUSTOM_PROCESSORS.items() if mode in spec.get("modes", [])]
