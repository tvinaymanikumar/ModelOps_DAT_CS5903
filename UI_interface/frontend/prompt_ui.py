"""Optional prompt editor (collapsed by default), one prompt per mode.

The edited prompt is kept under its own session key (saved_prompt_<mode>),
separate from the text box's widget key. Streamlit deletes a widget's state
whenever the widget is not drawn (for example after switching between
"No schema" and "With schema"), and it does not allow changing a widget's
value after the widget is drawn. Keeping a separate copy and changing it only
in callbacks fixes both: edits survive mode switches and Reset works.
"""
from __future__ import annotations

import streamlit as st

from config.task_config import TaskConfig, default_prompt
from processors.base_processor import render_prompt


def _saved_key(mode: str) -> str:
    return f"saved_prompt_{mode}"


def _widget_key(mode: str) -> str:
    return f"prompt_{mode}"


def _store(mode: str) -> None:
    """on_change: copy the text box into the saved prompt."""
    st.session_state[_saved_key(mode)] = st.session_state[_widget_key(mode)]


def _reset(mode: str) -> None:
    """on_click: runs before the widgets are drawn, so the text box may be changed here."""
    text = default_prompt(mode)
    st.session_state[_saved_key(mode)] = text
    st.session_state[_widget_key(mode)] = text


def current_prompt(mode: str) -> str:
    ss = st.session_state
    if _saved_key(mode) not in ss:
        ss[_saved_key(mode)] = default_prompt(mode)
    return ss[_saved_key(mode)]


def render_prompt_editor(task: TaskConfig) -> str:
    mode = task.mode
    saved = current_prompt(mode)
    wkey = _widget_key(mode)
    if wkey not in st.session_state:  # first draw, or Streamlit dropped it while hidden
        st.session_state[wkey] = saved
    edited = saved.strip() != default_prompt(mode).strip()
    with st.expander("Edit the prompt (optional)" + (" · edited" if edited else "")):
        st.text_area("Prompt", key=wkey, height=220, label_visibility="collapsed",
                     on_change=_store, args=(mode,),
                     help="With a schema, the schema block is added automatically. "
                          "Put {template} in the prompt to place the field template yourself. "
                          "Click outside the box (or press Ctrl+Enter) to apply your edit.")
        c1, c2 = st.columns([1, 3])
        c1.button("Reset prompt", key=f"reset_{wkey}", on_click=_reset, args=(mode,))
        c2.caption("The prompt as sent:")
        st.code(render_prompt(st.session_state[_saved_key(mode)], task.template, task.fields), language=None)
        if st.session_state.get("model_name") == "NuExtract3":
            st.caption("NuExtract3 is not driven by this prompt: it receives the field list as a template. "
                       "An edited prompt is sent to it as extra instructions.")
    return st.session_state[_saved_key(mode)]
