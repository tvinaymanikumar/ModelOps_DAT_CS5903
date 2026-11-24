"""Step 2 of the page: schema or no schema.

No schema   -> the model extracts every visible field (schema-free).
With schema -> the user types the fields they need; types and hints are
               optional. The field list generates the template sent to the
               model and the JSON Schema used to check the answer.
"""
from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd
import streamlit as st

from config.settings import PRESETS_DIR
from config.task_config import TaskConfig
from utils.file_utils import safe_stem
from validation.rules import rules_for_task
from validation.schema_validator import (FIELD_TYPES, clean_fields, fields_from_json, fields_from_names,
                                         schema_from_fields, template_from_fields)

NO_SCHEMA = "No schema"
WITH_SCHEMA = "With schema"
CHOICE_HELP = {
    NO_SCHEMA: "The model reads the document and returns every field it finds.",
    WITH_SCHEMA: "You list the fields you need. The result has exactly those fields.",
}


def _presets() -> dict[str, dict[str, Any]]:
    out = {}
    for f in sorted(PRESETS_DIR.glob("*.json")):
        try:
            out[f.stem] = json.loads(f.read_text(encoding="utf-8"))
        except ValueError:
            continue
    return out


def _queue_fields(fields: list[dict[str, Any]]) -> None:
    """Fill the field box on the next run (a widget's value cannot change after it is drawn)."""
    st.session_state["pending_fields"] = fields


def _apply_pending() -> None:
    ss = st.session_state
    fields = ss.pop("pending_fields", None)
    if fields is None:
        return
    ss["field_names"] = ", ".join(f["key"] for f in fields)
    ss["field_meta"] = {f["key"]: {"type": f["type"], "required": f["required"], "description": f["description"]}
                        for f in fields}


def _field_inputs() -> list[dict[str, Any]]:
    ss = st.session_state
    ss.setdefault("field_names", "")
    ss.setdefault("field_meta", {})
    _apply_pending()

    names = st.text_area("Fields you need", key="field_names", height=90,
                         placeholder="name, age, policy_number, claim_amount",
                         help="Separate field names with commas or new lines.")
    fields = fields_from_names(names)
    meta = ss["field_meta"]
    for f in fields:
        f.update(meta.get(f["key"], {}))
    fields, problems = clean_fields(fields)
    for p in problems:
        st.error(p)

    with st.expander("Field types and hints (optional)"):
        if not fields:
            st.caption("Type some field names first.")
        else:
            st.caption("Every field is text unless you change it. A hint tells the model where to look.")
            edited = st.data_editor(
                pd.DataFrame(fields, columns=["key", "type", "required", "description"]),
                key=f"types_{hash(tuple(f['key'] for f in fields))}",
                num_rows="fixed", hide_index=True, disabled=["key"],
                column_config={
                    "key": st.column_config.TextColumn("Field"),
                    "type": st.column_config.SelectboxColumn("Type", options=FIELD_TYPES, required=True),
                    "required": st.column_config.CheckboxColumn("Required",
                                                                help="Flag the result for review if this is empty."),
                    "description": st.column_config.TextColumn("Hint for the model"),
                })
            rows = edited.astype(object).where(pd.notna(edited), None).to_dict("records")
            fields, _ = clean_fields(rows)
            ss["field_meta"] = {f["key"]: {"type": f["type"], "required": f["required"],
                                           "description": f["description"]} for f in fields}

    with st.expander("Load or save a field list"):
        presets = _presets()
        if presets:
            c1, c2 = st.columns([3, 1])
            pick = c1.selectbox("Saved field lists", list(presets),
                                format_func=lambda k: presets[k].get("name", k))
            c2.markdown("&nbsp;")
            if c2.button("Load", width="stretch"):
                _queue_fields(fields_from_json(presets[pick]))
                st.rerun()
        up = st.file_uploader("Or import a JSON template or JSON Schema", type=["json"], key="schema_import")
        if up is not None and st.button("Use fields from this file"):
            try:
                _queue_fields(fields_from_json(json.loads(up.getvalue().decode("utf-8-sig"))))
                st.rerun()
            except (ValueError, UnicodeDecodeError) as exc:
                st.error(f"Could not read fields from {up.name}: {exc}")
        c1, c2 = st.columns([3, 1])
        pname = c1.text_input("Save the current fields as", placeholder="e.g. Motor claim form")
        c2.markdown("&nbsp;")
        if c2.button("Save", width="stretch", disabled=not (fields and pname.strip())):
            stem = safe_stem(re.sub(r"\s+", "_", pname.strip().lower()))
            (PRESETS_DIR / f"{stem}.json").write_text(json.dumps({"name": pname.strip(), "fields": fields}, indent=2),
                                                      encoding="utf-8")
            st.toast(f"Saved {pname.strip()}")

    if fields:
        with st.expander("Preview the output format"):
            st.code(json.dumps(template_from_fields(fields), indent=2), language="json")
    return fields


def render_schema_choice(transcription: bool) -> tuple[TaskConfig, list[str]]:
    """Return (task, problems that block Extract)."""
    if transcription:
        st.info("Handwriting transcription is on (sidebar). The result is plain text, so no schema is needed.")
        return TaskConfig(mode="transcription", name="Transcription", strict_keys=False,
                          rules=rules_for_task("transcription")), []

    choice = st.radio("Schema", [NO_SCHEMA, WITH_SCHEMA], horizontal=True, key="schema_choice",
                      label_visibility="collapsed")
    st.caption(CHOICE_HELP[choice])
    if choice == NO_SCHEMA:
        return TaskConfig(mode="schema_free", name="No schema", strict_keys=False,
                          rules=rules_for_task("schema_free")), []

    fields = _field_inputs()
    problems = [] if fields else ["add at least one field"]
    name = "Schema: " + ", ".join(f["key"] for f in fields[:4]) + ("…" if len(fields) > 4 else "")
    task = TaskConfig(mode="schema_based", name=name if fields else "Schema", fields=fields,
                      template=template_from_fields(fields),
                      schema=schema_from_fields(fields) if fields else None,
                      strict_keys=True, rules=rules_for_task("schema_based", fields))
    return task, problems
