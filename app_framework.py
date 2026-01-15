"""
Streamlit app scaffold with explicit finite state machine (FSM) for deterministic workflow.
All interactions live in the sidebar. Plotly is used for all plots, with PNG (kaleido) export fallback.

States:
- EMPTY: no data loaded
- DATA_LOADED: data present, waiting for column selection
- COLUMN_SELECTED: column chosen, start/end not set
- START_SET: start index chosen, end pending
- END_SET: start/end chosen

Transitions are explicit and centralized; UI visibility derives only from workflow_state.
"""

import io
import json
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ============== #
# Constants / FSM
# ============== #
STATE_EMPTY = "EMPTY"
STATE_DATA_LOADED = "DATA_LOADED"
STATE_COLUMN_SELECTED = "COLUMN_SELECTED"
STATE_START_SET = "START_SET"
STATE_END_SET = "END_SET"

STATE_ORDER = [STATE_EMPTY, STATE_DATA_LOADED, STATE_COLUMN_SELECTED, STATE_START_SET, STATE_END_SET]


# ==========================
# State transition utilities
# ==========================
def ensure_session_defaults() -> None:
    defaults = {
        "workflow_state": STATE_EMPTY,
        "df": None,
        "columns": [],
        "selected_column": None,
        "start_index": None,
        "end_index": None,
        "analysis_name": "provide_name_of_current_analysis",
        "processed_upload_sig": None,
        "processed_zip_sig": None,
        "export_in_progress": False,
        "original_filename": None,
        "export_timestamp": "not yet exported",
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def reset_for_new_upload(df: pd.DataFrame, columns: List[str]) -> None:
    """Clear downstream inputs and set state to DATA_LOADED after a fresh upload."""
    st.session_state["df"] = df
    st.session_state["columns"] = columns
    st.session_state["selected_column"] = None
    st.session_state["start_index"] = None
    st.session_state["end_index"] = None
    st.session_state["analysis_name"] = "provide_name_for_current_analysis"
    transition_to_data_loaded()


def transition_to_data_loaded() -> None:
    st.session_state["workflow_state"] = STATE_DATA_LOADED


def transition_on_column_change(selected: Optional[str]) -> None:
    st.session_state["selected_column"] = selected
    st.session_state["start_index"] = None
    st.session_state["end_index"] = None
    st.session_state["scroll_target"] = None
    if selected:
        st.session_state["workflow_state"] = STATE_COLUMN_SELECTED
    else:
        st.session_state["workflow_state"] = STATE_DATA_LOADED


def validate_and_transition_start(start_val: Optional[int]) -> None:
    df = st.session_state.get("df")
    col = st.session_state.get("selected_column")
    if df is None or col is None:
        return

    if start_val is None or start_val < 0 or start_val >= len(df):
        st.session_state["start_index"] = None
        st.session_state["end_index"] = None
        st.session_state["workflow_state"] = STATE_COLUMN_SELECTED
        return

    st.session_state["start_index"] = int(start_val)
    st.session_state["end_index"] = None
    st.session_state["workflow_state"] = STATE_START_SET


def validate_and_transition_end(end_val: Optional[int]) -> None:
    df = st.session_state.get("df")
    start_val = st.session_state.get("start_index")
    if df is None or start_val is None:
        return

    if end_val is None or end_val < start_val or end_val >= len(df):
        st.session_state["end_index"] = None
        st.session_state["workflow_state"] = STATE_START_SET
        return

    st.session_state["end_index"] = int(end_val)
    st.session_state["workflow_state"] = STATE_END_SET


# ==================
# Data load helpers
# ==================
def read_csv_to_df(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    # Ensure a time column exists; if missing, synthesize incremental time
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
    else:
        base = datetime(2026, 1, 1)
        df["time"] = [base + timedelta(seconds=i) for i in range(len(df))]
    df = df.reset_index(drop=True)
    df["index_col"] = np.arange(len(df))
    return df


def infer_numeric_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != "index_col"]


# =============
# Plot helpers
# =============
def _base_plot(df: pd.DataFrame, col: str, mask: Optional[pd.Series] = None, title: str = ""):
    data = df if mask is None else df.loc[mask].reset_index(drop=True)
    fig = px.line(
        data,
        x="time",
        y=col,
        hover_data={"index_col": True, "time": "|%Y-%m-%d %H:%M:%S", col: True},
        title=title,
    )
    fig.update_traces(
        hovertemplate="index=%{customdata[0]}<br>time=%{x|%Y-%m-%d %H:%M:%S}<br>value=%{y}"
    )
    fig.update_layout(xaxis_title="time", yaxis_title=col)
    return fig


def build_plot_1(df: pd.DataFrame, col: str):
    return _base_plot(df, col, title="Plot 1: full series")


def build_plot_2(df: pd.DataFrame, col: str, start: int):
    mask = df["index_col"] >= start
    return _base_plot(df, col, mask=mask, title="Plot 2: start -> end")


def build_plot_3(df: pd.DataFrame, col: str, start: int, end: int):
    mask = (df["index_col"] >= start) & (df["index_col"] <= end)
    return _base_plot(df, col, mask=mask, title="Plot 3: bounded interval")


# ===============
# Export helpers
# ===============
def save_fig_html(fig, path: Path) -> None:
    fig.write_html(path, include_plotlyjs="cdn", full_html=True)


def save_fig_png(fig, path: Path) -> Optional[str]:
    try:
        fig.write_image(path, format="png", scale=2)
        return None
    except Exception as exc:  # noqa: BLE001
        return str(exc)


def export_analysis_to_zip(
    analysis_name: str,
    df: pd.DataFrame,
    config: Dict[str, object],
    fig1,
    fig2=None,
    fig3=None,
    spinner_box=None,
) -> Tuple[bytes, List[str]]:
    """Create a zip containing analysis folder with config, data, and plots."""
    warnings: List[str] = []
    base_dir = Path(".")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        # config
        zf.writestr("config.json", json.dumps(config, indent=2))
        # data
        data_bytes = df.to_csv(index=False).encode("utf-8")
        zf.writestr("data.csv", data_bytes)
        # plots html
        figs = [fig1, fig2, fig3]
        n_figs = len(figs)
        for idx, fig in enumerate(figs, start=1):
            if fig is None:
                continue
            show_status(spinner_box, f'Exporting plot [{idx}/{n_figs}] ...')
            html_bytes = fig.to_html(full_html=True, include_plotlyjs="cdn").encode("utf-8")
            zf.writestr(f"plot{idx}.html", html_bytes)
            # png
            try:
                png_bytes = fig.to_image(format="png", scale=2)
                zf.writestr(f"plot{idx}.png", png_bytes)
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"Plot {idx} PNG not saved (install kaleido): {exc}")
    return buf.getvalue(), warnings


def load_analysis_from_zip(file) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Load analysis zip (config.json + data.csv)."""
    with zipfile.ZipFile(file, "r") as zf:
        config = json.loads(zf.read("config.json"))
        data_bytes = zf.read("data.csv")
    df = pd.read_csv(io.BytesIO(data_bytes))
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.reset_index(drop=True)
    df["index_col"] = np.arange(len(df))
    return df, config


# =======================
# UI rendering / anchors
# =======================
def render_sidebar(df: Optional[pd.DataFrame], numeric_cols: List[str]) -> Tuple[Optional[str], Optional[int], Optional[int]]:
    st.sidebar.header("Data source")
    uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"], key="sidebar_csv")
    loaded_zip = st.sidebar.file_uploader("Load analysis (zip)", type=["zip"], key="sidebar_zip")
    load_zip_clicked = st.sidebar.button("Load analysis zip", key="load_zip_btn")

    if loaded_zip and load_zip_clicked:
        try:
            sig = (loaded_zip.name, loaded_zip.size)
            if sig != st.session_state.get("processed_zip_sig"):
                df_loaded, cfg = load_analysis_from_zip(loaded_zip)
                cols = infer_numeric_columns(df_loaded)
                reset_for_new_upload(df_loaded, cols)
                st.session_state["selected_column"] = cfg.get("selected_column")
                st.session_state["start_index"] = cfg.get("start_index")
                st.session_state["end_index"] = cfg.get("end_index")
                st.session_state["workflow_state"] = cfg.get("workflow_state", STATE_DATA_LOADED)
                st.session_state["analysis_name"] = cfg.get("analysis_name", "provide_name_of_current_analysis")
                st.session_state["original_filename"] = cfg.get("original_filename", loaded_zip.name)
                st.session_state["export_timestamp"] = cfg.get("export_timestamp", "not yet exported")
                st.session_state["processed_zip_sig"] = sig
                if hasattr(st, "rerun"):
                    st.rerun()
        except Exception as exc:  # noqa: BLE001
            st.error(f"Failed to load analysis: {exc}")

    if uploaded:
        try:
            sig = (uploaded.name, uploaded.size)
            if sig != st.session_state.get("processed_upload_sig"):
                df_new = read_csv_to_df(uploaded)
                cols = infer_numeric_columns(df_new)
                reset_for_new_upload(df_new, cols)
                st.session_state["processed_upload_sig"] = sig
                st.session_state["original_filename"] = uploaded.name
                st.session_state["export_timestamp"] = "not yet exported"
                if hasattr(st, "rerun"):
                    st.rerun()
        except Exception as exc:  # noqa: BLE001
            st.error(f"Failed to read CSV: {exc}")

    st.sidebar.markdown("---")
    st.sidebar.header("Inputs")
    if st.session_state["workflow_state"] == STATE_EMPTY:
        with st.sidebar:
            with st.spinner("Waiting for data... Upload a CSV or load an analysis zip to begin."):
                st.write("")
        return None, None, None

    sel_col = None
    start_val = None
    end_val = None

    if st.session_state["workflow_state"] in {STATE_DATA_LOADED, STATE_COLUMN_SELECTED, STATE_START_SET, STATE_END_SET}:
        # Avoid on_change firing before options render; we rely on explicit compare below.
        sel_col = st.sidebar.selectbox(
            "Select numeric column",
            [""] + numeric_cols,
            index=(numeric_cols.index(st.session_state["selected_column"]) + 1) if st.session_state["selected_column"] in numeric_cols else 0,
            key="select_column",
        )
        if sel_col == "":
            sel_col = None
        if sel_col != st.session_state.get("selected_column"):
            transition_on_column_change(sel_col)

    if st.session_state["workflow_state"] in {STATE_COLUMN_SELECTED, STATE_START_SET, STATE_END_SET} and df is not None and st.session_state.get("selected_column"):
        max_idx = len(df) - 1
        with st.sidebar.form("start_form"):
            st.markdown("**Step 2: set start index**")
            start_val = st.number_input(
                "Start index",
                min_value=0,
                max_value=max_idx,
                value=st.session_state["start_index"] if st.session_state["start_index"] is not None else 0,
                key="start_index_input",
            )
            start_submit = st.form_submit_button("Set start")
        if start_submit:
            validate_and_transition_start(start_val)

    if st.session_state["workflow_state"] in {STATE_START_SET, STATE_END_SET} and df is not None and st.session_state.get("start_index") is not None and st.session_state.get("selected_column"):
        max_idx = len(df) - 1
        min_val = st.session_state["start_index"]
        with st.sidebar.form("end_form"):
            st.markdown("**Step 3: set end index**")
            end_val = st.number_input(
                "End index",
                min_value=min_val,
                max_value=max_idx,
                value=st.session_state["end_index"] if st.session_state["end_index"] is not None else min_val,
                key="end_index_input",
            )
            end_submit = st.form_submit_button("Set end")
        if end_submit:
            validate_and_transition_end(end_val)

    return sel_col, start_val, end_val


def render_main(df: Optional[pd.DataFrame], fig1, fig2, fig3) -> None:
    st.title("FSM Template with Plotly")
    st.caption(f"Workflow state: {st.session_state.get('workflow_state')}")
    ts_raw = st.session_state.get("export_timestamp")
    ts_fmt = "not yet exported"
    if ts_raw and ts_raw != "not yet exported":
        try:
            parsed = datetime.fromisoformat(str(ts_raw).replace("Z", "+00:00"))
            local_dt = parsed.astimezone()
            ts_fmt = local_dt.strftime("%d-%m-%Y %H:%M:%S %Z%z")
        except Exception:
            ts_fmt = str(ts_raw)
    st.caption(
        f"Original file: {st.session_state.get('original_filename') or '-'} | "
        f"Export time: {ts_fmt}"
    )

    if fig1 is not None:
        st.plotly_chart(fig1, use_container_width=True)

    if fig2 is not None and st.session_state["workflow_state"] in {STATE_START_SET, STATE_END_SET}:
        st.plotly_chart(fig2, use_container_width=True)

    if fig3 is not None and st.session_state["workflow_state"] == STATE_END_SET:
        st.plotly_chart(fig3, use_container_width=True)

def show_status(spinner_box, phase: str) -> None:
    if spinner_box is None:
        return
    spinner_box.markdown(
        """
        <div style="display:flex;flex-direction:column;gap:6px;">
            <div style="display:flex;align-items:center;gap:8px;">
            <div class="spinner" style="
                    width:16px;height:16px;
                    border:3px solid rgba(234,109,8,0.2);
                    border-top-color:#EA6D08;
                    border-radius:50%;
                    animation:spin 0.8s linear infinite;"></div>
            <span><strong>Exporting…</strong><br>-> PHASE_TXT</span>
            </div>
            <div style="font-size:12px;color:#999;">
            </div>
        </div>
        <style>
        @keyframes spin { from {transform:rotate(0deg);} to {transform:rotate(360deg);} }
        </style>
        """.replace("PHASE_TXT", phase),
        unsafe_allow_html=True,
    )

def render_export_section(df: pd.DataFrame, fig1, fig2, fig3) -> None:
    st.sidebar.markdown("---")
    st.sidebar.header("Export")
    st.session_state["analysis_name"] = st.sidebar.text_input(
        "Analysis name",
        value=st.session_state.get("analysis_name", "provide_name_of_current_analysis"),
    )
    export_btn = st.sidebar.button("Export analysis", key="export_btn")

    if export_btn:
        spinner_box = st.sidebar.empty()

        show_status(spinner_box, "Writing config & data")
        export_ts = datetime.now().astimezone().isoformat()
        st.session_state["export_timestamp"] = export_ts
        config = {
            "analysis_name": st.session_state["analysis_name"],
            "workflow_state": st.session_state["workflow_state"],
            "selected_column": st.session_state["selected_column"],
            "start_index": st.session_state["start_index"],
            "end_index": st.session_state["end_index"],
            "row_count": len(df),
            "columns": list(df.columns),
            "original_filename": st.session_state.get("original_filename"),
            "export_timestamp": export_ts,
        }
        # Plots phase
        show_status(spinner_box, "Exporting plots")
        data_bytes, warnings = export_analysis_to_zip(
            st.session_state["analysis_name"],
            df,
            config,
            fig1,
            fig2,
            fig3,
            spinner_box=spinner_box,
        )
        show_status(spinner_box, "Creating zip")
        spinner_box.empty()
        st.sidebar.markdown(
            """
            <style>
            /* Style and pulse the export download button */
            div[data-testid="stDownloadButton"] > button {
                background-color: #EA6D08 !important;
                color: #ffffff !important;
                border: 0 !important;
                font-weight: 600;
                animation: pulse-export 1.4s ease-in-out 2;
            }
            div[data-testid="stDownloadButton"] > button:hover {
                filter: brightness(1.05);
            }
            @keyframes pulse-export {
                0% { box-shadow: 0 0 0 0 rgba(234, 109, 8, 0.6); }
                70% { box-shadow: 0 0 0 12px rgba(234, 109, 8, 0); }
                100% { box-shadow: 0 0 0 0 rgba(234, 109, 8, 0); }
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        st.sidebar.download_button(
            label="Download export zip",
            data=data_bytes,
            file_name=f"{st.session_state['analysis_name']}.zip",
            mime="application/zip",
        )
        if warnings:
            st.sidebar.warning("\n".join(warnings))


# =====
# Main
# =====
def main():
    st.set_page_config(page_title="FSM Plotly Scaffold", layout="wide")
    ensure_session_defaults()

    df = st.session_state.get("df")
    numeric_cols = infer_numeric_columns(df) if df is not None else []

    sel_col, start_val, end_val = render_sidebar(df, numeric_cols)

    # Transition after processing sidebar selections
    if st.session_state["workflow_state"] == STATE_DATA_LOADED and st.session_state.get("selected_column"):
        st.session_state["workflow_state"] = STATE_COLUMN_SELECTED
    if st.session_state["workflow_state"] == STATE_COLUMN_SELECTED:
        if st.session_state.get("start_index") is not None:
            st.session_state["workflow_state"] = STATE_START_SET
    if st.session_state["workflow_state"] == STATE_START_SET:
        if st.session_state.get("end_index") is not None:
            st.session_state["workflow_state"] = STATE_END_SET

    # Build plots based on state
    fig1 = fig2 = fig3 = None
    if st.session_state["workflow_state"] in {STATE_COLUMN_SELECTED, STATE_START_SET, STATE_END_SET} and df is not None and st.session_state.get("selected_column"):
        fig1 = build_plot_1(df, st.session_state["selected_column"])
    if st.session_state["workflow_state"] in {STATE_START_SET, STATE_END_SET} and df is not None and st.session_state.get("selected_column") and st.session_state.get("start_index") is not None:
        fig2 = build_plot_2(df, st.session_state["selected_column"], st.session_state["start_index"])
    if st.session_state["workflow_state"] == STATE_END_SET and df is not None and st.session_state.get("selected_column") and st.session_state.get("start_index") is not None and st.session_state.get("end_index") is not None:
        fig3 = build_plot_3(df, st.session_state["selected_column"], st.session_state["start_index"], st.session_state["end_index"])

    render_main(df, fig1, fig2, fig3)

    if st.session_state["workflow_state"] == STATE_END_SET and df is not None:
        render_export_section(df, fig1, fig2, fig3)


if __name__ == "__main__":
    main()
