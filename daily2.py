# daily2.py
# Streamlit Culture Rater with batching via URL query param: ?batch=1
#
# What batching does here:
# - Participants still upload ONE CSV (your merged_dialogues.csv).
# - The app shows ONLY a slice of items for the chosen batch.
# - Downloaded CSV contains ONLY the annotated rows for that batch (full original columns, subset of rows).
#
# You control which batch they see by giving them a link like:
#   https://YOURAPP.streamlit.app/?batch=1
#   https://YOURAPP.streamlit.app/?batch=2
#
# Defaults:
# - BATCH_SIZE_ITEMS = 40  (items = (row, model_col) pairs, because your pool mixes models)
# - If batch is missing/invalid, it falls back to 1.
#
# IMPORTANT:
# Because your "item_pool" mixes ALL model columns, one "item" = one dialogue text from one model column.
# If you want 40 "conversations" regardless of model, keep it as-is (40 items).
# If you want 40 "rows" (and rate all 5 model outputs per row), tell me and I'll rewrite the UI.

import random
import csv
import math
from typing import Tuple, List, Dict, Any

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Culture Rater", layout="centered")

MODEL_TEXT_COLS = [
    "Chatgpt 5.2",
    "Gemini Pro",
    "Qwen 3-Max",
    "DeepSeek v3.2",
    "Mistral Large",
]

RATING_PREFIX = "confidence_rating_"
NOTE_PREFIX = "rating_note_"

# --- Batch config ---
BATCH_SIZE_ITEMS = 40  # number of "items" per batch (each item is one model's dialogue text)
BATCH_PARAM_NAME = "batch"


# =========================
# Helpers
# =========================
def clean_cell(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    return "" if s.lower() == "nan" else s


def slugify_col(col_name: str) -> str:
    s = "".join(ch if ch.isalnum() else "_" for ch in col_name.strip())
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")


def robust_read_csv(file_like) -> Tuple[pd.DataFrame, str]:
    for sep in [",", ";", "\t", "|"]:
        try:
            try:
                file_like.seek(0)
            except Exception:
                pass

            df = pd.read_csv(
                file_like,
                sep=sep,
                engine="python",
                quoting=csv.QUOTE_MINIMAL,
            )
            if len(df.columns) == 1 and sep != "|":
                continue
            return df, sep
        except Exception:
            continue

    try:
        try:
            file_like.seek(0)
        except Exception:
            pass

        df = pd.read_csv(
            file_like,
            sep=",",
            engine="python",
            on_bad_lines="skip",
            quoting=csv.QUOTE_MINIMAL,
        )
        return df, ","
    except Exception as e:
        st.error(f"Could not read CSV.\n\nError: {e}")
        st.stop()


def ensure_model_columns_exist(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in MODEL_TEXT_COLS if c not in df.columns]
    if missing:
        st.error(
            "Some required text columns are missing.\n\n"
            f"Missing: {missing}\n\n"
            f"Available columns: {list(df.columns)}"
        )
        st.stop()

    for c in MODEL_TEXT_COLS:
        suf = slugify_col(c)
        rating_col = f"{RATING_PREFIX}{suf}"
        note_col = f"{NOTE_PREFIX}{suf}"
        if rating_col not in df.columns:
            df[rating_col] = ""
        if note_col not in df.columns:
            df[note_col] = ""
    return df


def build_item_pool(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Create ONE mixed pool of (row_id, model_col, text)."""
    pool: List[Dict[str, Any]] = []
    for model_col in MODEL_TEXT_COLS:
        for row_id, val in df[model_col].items():
            t = clean_cell(val)
            if t:
                pool.append({"row_id": int(row_id), "model_col": model_col, "text": t})
    if not pool:
        st.error("No non-empty dialogs found across the specified model columns.")
        st.stop()
    return pool


def write_rating_to_df(df: pd.DataFrame, row_id: int, model_col: str, rating: str, note: str) -> None:
    suf = slugify_col(model_col)
    rating_col = f"{RATING_PREFIX}{suf}"
    note_col = f"{NOTE_PREFIX}{suf}"
    df.at[row_id, rating_col] = clean_cell(rating)
    df.at[row_id, note_col] = clean_cell(note)


def clamp_pos(n_items: int) -> None:
    st.session_state.pos = max(0, min(st.session_state.pos, max(0, n_items - 1)))


def get_batch_id() -> int:
    """Read ?batch= from URL. Defaults to 1."""
    try:
        # streamlit query params acts like dict[str, str|list[str]]
        raw = st.query_params.get(BATCH_PARAM_NAME, "1")
        if isinstance(raw, list):
            raw = raw[0] if raw else "1"
        bid = int(str(raw).strip())
        return 1 if bid < 1 else bid
    except Exception:
        return 1


def compute_batch_bounds(n_items: int, batch_id: int, batch_size: int) -> Tuple[int, int, int]:
    """Return (start, end_exclusive, n_batches)."""
    n_batches = max(1, math.ceil(n_items / batch_size))
    batch_id = max(1, min(batch_id, n_batches))
    start = (batch_id - 1) * batch_size
    end = min(start + batch_size, n_items)
    return start, end, n_batches


def get_batch_signature(file_signature: str, batch_id: int) -> str:
    return f"{file_signature}::batch={batch_id}"


# =========================
# SIDEBAR
# =========================
st.sidebar.title("Setup")
rater_name = st.sidebar.text_input("Rater name (required)", value="")
uploaded_file = st.sidebar.file_uploader("Upload your CSV", type=["csv"])

batch_id = get_batch_id()

if not rater_name.strip():
    st.info("Enter your name in the sidebar to start.")
    st.stop()

if uploaded_file is None:
    st.info("Upload your CSV in the sidebar to start.")
    st.stop()

# =========================
# LOAD ONCE PER FILE (persist df + pool)
# =========================
file_signature = f"{uploaded_file.name}-{uploaded_file.size}"

# Load the whole DF once per uploaded file
if st.session_state.get("file_signature") != file_signature:
    df0, sep0 = robust_read_csv(uploaded_file)
    df0 = ensure_model_columns_exist(df0)

    st.session_state.file_signature = file_signature
    st.session_state.source_label = f"Uploaded: {uploaded_file.name}"
    st.session_state.sep_used = sep0
    st.session_state.df_full = df0
    st.session_state.item_pool_full = build_item_pool(df0)

# For convenience
df_full: pd.DataFrame = st.session_state.df_full
sep_used: str = st.session_state.sep_used
source_label: str = st.session_state.source_label
item_pool_full: List[Dict[str, Any]] = st.session_state.item_pool_full

# =========================
# BATCH SLICE (pool-level)
# =========================
start, end, n_batches = compute_batch_bounds(len(item_pool_full), batch_id, BATCH_SIZE_ITEMS)
item_pool = item_pool_full[start:end]
n_items = len(item_pool)

if n_items == 0:
    st.error("This batch has no items. Please check the batch number in the URL.")
    st.stop()

# Reset per-batch ordering/position if (file,batch) changed
batch_signature = get_batch_signature(file_signature, batch_id)
if st.session_state.get("batch_signature") != batch_signature:
    st.session_state.batch_signature = batch_signature
    st.session_state.order = list(range(n_items))
    random.shuffle(st.session_state.order)
    st.session_state.pos = 0

# Sidebar controls for this batch
st.sidebar.subheader("Batch")
st.sidebar.markdown(f"**Batch:** {batch_id} / {n_batches}")
st.sidebar.markdown(f"**Items in this batch:** {n_items} (pool slice)")

st.sidebar.subheader("Controls")
if st.sidebar.button("Reshuffle order (this batch)"):
    st.session_state.order = list(range(n_items))
    random.shuffle(st.session_state.order)
    st.session_state.pos = 0
    st.rerun()

# =========================
# MAIN UI
# =========================
st.title("Culture Rater")
st.caption(
    " • ".join(
        [
            source_label,
            f"Batch {batch_id}/{n_batches}",
            f"Items in this batch: {n_items}",
            f"Rater: {rater_name}",
        ]
    )
)

with st.expander("A/B/C/D meaning (confidence it is my country)", expanded=True):
    st.markdown("- **A** — Very sure: Unmistakably my country’s culture.")
    st.markdown("- **B** — Fairly sure: Mostly fits my country; a few details could fit elsewhere.")
    st.markdown("- **C** — Not sure: Could be many countries; weak or generic cultural cues.")
    st.markdown("- **D** — Sure it is NOT mine: Clearly mismatched; points elsewhere or feels wrong.")

if "pos" not in st.session_state:
    st.session_state.pos = 0

clamp_pos(n_items)

idx_in_pool = st.session_state.order[st.session_state.pos]
item = item_pool[idx_in_pool]

row_id = item["row_id"]
model_col = item["model_col"]
text = item["text"]

suf = slugify_col(model_col)
rating_col = f"{RATING_PREFIX}{suf}"
note_col = f"{NOTE_PREFIX}{suf}"

existing_note = clean_cell(df_full.at[row_id, note_col])
existing_rating = clean_cell(df_full.at[row_id, rating_col])

st.write(f"Item {st.session_state.pos + 1} / {n_items}")
st.caption(f"Model column: **{model_col}**  •  Saved rating: **{existing_rating or '-'}**")

st.markdown("---")
st.markdown(
    f"<div style='padding:14px;border-radius:10px;border:1px solid #ddd;background:#fafafa;white-space:pre-wrap;'>{text}</div>",
    unsafe_allow_html=True,
)
st.markdown("---")

note_key = f"note_{batch_id}_{row_id}_{suf}"
sel_key = f"selected_rating_{batch_id}_{row_id}_{suf}"

# init widgets once
if note_key not in st.session_state:
    st.session_state[note_key] = existing_note
if sel_key not in st.session_state:
    st.session_state[sel_key] = existing_rating

st.text_area("Note (optional)", key=note_key, height=90)

st.subheader("Select rating (A–D)")
st.radio(
    "Choose one:",
    options=["", "A", "B", "C", "D"],
    format_func=lambda x: "— (not selected)" if x == "" else x,
    key=sel_key,
    horizontal=True,
)
st.caption(f"Current selection (not saved yet): **{st.session_state[sel_key] or '-'}**")

# =========================
# NAV
# =========================
nav_cols = st.columns(2)

with nav_cols[0]:
    if st.button("Prev", use_container_width=True):
        st.session_state.pos -= 1
        clamp_pos(n_items)
        st.rerun()

with nav_cols[1]:
    if st.button("Next (save & go)", use_container_width=True):
        if not st.session_state[sel_key]:
            st.warning("Please select A/B/C/D before going to next.")
            st.stop()

        write_rating_to_df(
            df_full,
            row_id,
            model_col,
            st.session_state[sel_key],
            st.session_state[note_key],
        )

        if st.session_state.pos < n_items - 1:
            st.session_state.pos += 1
        clamp_pos(n_items)
        st.rerun()

# =========================
# EXPORT (BATCH ONLY)
# =========================
st.subheader("Export updated CSV (this batch only)")

# collect unique row_ids present in this batch, then export those rows only
batch_row_ids = sorted({it["row_id"] for it in item_pool})
df_batch_rows = df_full.loc[batch_row_ids].copy()

default_name = (
    f"{uploaded_file.name.replace('.csv','')}"
    f"_batch_{batch_id:02d}"
    f"_rated_{rater_name.strip().replace(' ','_')}.csv"
)

export_name = st.text_input("Download filename", value=default_name)

csv_bytes = df_batch_rows.to_csv(index=False, sep=sep_used, quoting=csv.QUOTE_MINIMAL).encode("utf-8")
st.download_button(
    label="Download updated CSV",
    data=csv_bytes,
    file_name=export_name,
    mime="text/csv",
)

st.info(
    "Tip: After finishing this batch, download the updated CSV and upload it using the submission form.\n\n"
    f"Batch: {batch_id} / {n_batches}"
)
