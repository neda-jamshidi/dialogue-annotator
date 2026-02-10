# daily2.py
# Culture Rater (Streamlit)
# - Rates EACH dialog text across multiple model columns (A/B/C/D + optional note)
# - Robust CSV reading (auto-detect separator)
# - Persists per-upload session (shuffle order, resume position)
# - Adds rating/note columns if missing
# - Includes optional "Skip" (doesn't force a rating) + "Save" separate from Next

import csv
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

# -----------------------
# CONFIG
# -----------------------
st.set_page_config(page_title="Culture Rater", layout="centered")

MODEL_TEXT_COLS = [
    "Chatgpt 5.2",
    "Gemini Pro",
    "Qwen 3-Max",
    "DeepSeek v3.2",
    "Mistral Large",
]

# Optional metadata columns to show if present (won't fail if missing)
META_COLS = ["Type of Prompt", "Prompt", "reference"]

RATING_PREFIX = "confidence_rating_"  # will create confidence_rating_<slug(model)>
NOTE_PREFIX = "rating_note_"          # will create rating_note_<slug(model)>

RATINGS = ["A", "B", "C", "D"]
RATING_LABELS = {
    "A": "Very sure (my country)",
    "B": "Fairly sure",
    "C": "Not sure",
    "D": "Sure NOT mine",
}


# -----------------------
# HELPERS
# -----------------------
def clean_cell(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    return "" if s.lower() == "nan" else s


def slugify_col(col_name: str) -> str:
    s = "".join(ch if ch.isalnum() else "_" for ch in col_name.strip())
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_").lower()


def robust_read_csv(file_like) -> Tuple[pd.DataFrame, str]:
    """
    Try common separators. Falls back to comma with on_bad_lines='skip'.
    Returns (df, sep_used).
    """
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
            # If it collapsed into 1 col for common seps, keep trying (except '|')
            if len(df.columns) == 1 and sep != "|":
                continue
            return df, sep
        except Exception:
            continue

    # fallback
    try:
        try:
            file_like.seek(0)
        except Exception:
            pass

        df = pd.read_csv(
            file_like,
            sep=",",
            engine="python",
            quoting=csv.QUOTE_MINIMAL,
            on_bad_lines="skip",
        )
        return df, ","
    except Exception as e:
        st.error(f"Could not read CSV.\n\nError: {e}")
        st.stop()


def ensure_required_text_cols(df: pd.DataFrame) -> None:
    missing = [c for c in MODEL_TEXT_COLS if c not in df.columns]
    if missing:
        st.error(
            "Some required model text columns are missing.\n\n"
            f"Missing: {missing}\n\n"
            f"Available columns: {list(df.columns)}"
        )
        st.stop()


def ensure_rating_columns(df: pd.DataFrame) -> None:
    """
    For each model column, ensure rating and note columns exist.
    """
    for c in MODEL_TEXT_COLS:
        suf = slugify_col(c)
        rcol = f"{RATING_PREFIX}{suf}"
        ncol = f"{NOTE_PREFIX}{suf}"
        if rcol not in df.columns:
            df[rcol] = ""
        if ncol not in df.columns:
            df[ncol] = ""


@dataclass(frozen=True)
class Item:
    row_id: int
    model_col: str
    text: str


def build_item_pool(df: pd.DataFrame) -> List[Item]:
    pool: List[Item] = []
    for model_col in MODEL_TEXT_COLS:
        for row_id, val in df[model_col].items():
            t = clean_cell(val)
            if t:
                pool.append(Item(row_id=row_id, model_col=model_col, text=t))

    if not pool:
        st.error("No non-empty dialogs found across the specified model columns.")
        st.stop()

    return pool


def get_rating_cols(model_col: str) -> Tuple[str, str]:
    suf = slugify_col(model_col)
    return f"{RATING_PREFIX}{suf}", f"{NOTE_PREFIX}{suf}"


def write_rating(df: pd.DataFrame, row_id: int, model_col: str, rating: str, note: str) -> None:
    rcol, ncol = get_rating_cols(model_col)
    df.at[row_id, rcol] = clean_cell(rating)
    df.at[row_id, ncol] = clean_cell(note)


def clamp_pos(pos: int, n: int) -> int:
    if n <= 0:
        return 0
    return max(0, min(pos, n - 1))


def progress_counts(df: pd.DataFrame, pool: List[Item]) -> Tuple[int, int]:
    """
    Returns (rated_count, total_items) using non-empty rating cells.
    """
    rated = 0
    for it in pool:
        rcol, _ = get_rating_cols(it.model_col)
        if clean_cell(df.at[it.row_id, rcol]):
            rated += 1
    return rated, len(pool)


# -----------------------
# SIDEBAR: SETUP
# -----------------------
st.sidebar.title("Setup")
rater_name = st.sidebar.text_input("Rater name (required)", value="")
uploaded_file = st.sidebar.file_uploader("Upload your CSV", type=["csv"])

if not rater_name.strip():
    st.info("Enter your name in the sidebar to start.")
    st.stop()

if uploaded_file is None:
    st.info("Upload your CSV in the sidebar to start.")
    st.stop()

file_signature = f"{uploaded_file.name}-{uploaded_file.size}"

# -----------------------
# LOAD / INIT SESSION
# -----------------------
if st.session_state.get("file_signature") != file_signature:
    df0, sep0 = robust_read_csv(uploaded_file)
    ensure_required_text_cols(df0)
    ensure_rating_columns(df0)

    pool0 = build_item_pool(df0)

    order0 = list(range(len(pool0)))
    random.shuffle(order0)

    st.session_state.file_signature = file_signature
    st.session_state.source_label = f"Uploaded: {uploaded_file.name}"
    st.session_state.sep_used = sep0
    st.session_state.df = df0
    st.session_state.pool = pool0
    st.session_state.order = order0
    st.session_state.pos = 0

df: pd.DataFrame = st.session_state.df
sep_used: str = st.session_state.sep_used
source_label: str = st.session_state.source_label
pool: List[Item] = st.session_state.pool
order: List[int] = st.session_state.order

# -----------------------
# SIDEBAR: CONTROLS
# -----------------------
st.sidebar.subheader("Controls")

if st.sidebar.button("Reshuffle order"):
    st.session_state.order = list(range(len(pool)))
    random.shuffle(st.session_state.order)
    st.session_state.pos = 0
    st.rerun()

rated_n, total_n = progress_counts(df, pool)
st.sidebar.metric("Progress", f"{rated_n}/{total_n}")

jump_to = st.sidebar.number_input(
    "Jump to item #",
    min_value=1,
    max_value=max(1, total_n),
    value=min(st.session_state.pos + 1, max(1, total_n)),
    step=1,
)
if st.sidebar.button("Go"):
    st.session_state.pos = clamp_pos(int(jump_to) - 1, total_n)
    st.rerun()

# -----------------------
# MAIN UI
# -----------------------
st.title("Culture Rater")
st.caption(" • ".join([source_label, f"Items: {len(pool)}", f"Rater: {rater_name.strip()}"]))

with st.expander("A/B/C/D meaning (confidence it is my country)", expanded=True):
    st.markdown("- **A** — Very sure: unmistakably my country’s culture.")
    st.markdown("- **B** — Fairly sure: mostly fits my country; a few details could fit elsewhere.")
    st.markdown("- **C** — Not sure: could be many countries; weak or generic cultural cues.")
    st.markdown("- **D** — Sure it is NOT mine: clearly mismatched; points elsewhere or feels wrong.")

# position
st.session_state.pos = clamp_pos(st.session_state.pos, len(pool))
idx_in_pool = order[st.session_state.pos]
item = pool[idx_in_pool]

row_id = item.row_id
model_col = item.model_col
text = item.text

rcol, ncol = get_rating_cols(model_col)
existing_rating = clean_cell(df.at[row_id, rcol])
existing_note = clean_cell(df.at[row_id, ncol])

st.write(f"Item {st.session_state.pos + 1} / {len(pool)}")
st.caption(f"Model column: **{model_col}**  •  Saved rating: **{existing_rating or '-'}**")

# Show metadata (if exists)
meta_lines = []
for mc in META_COLS:
    if mc in df.columns:
        v = clean_cell(df.at[row_id, mc])
        if v:
            meta_lines.append((mc, v))

if meta_lines:
    with st.expander("Metadata", expanded=False):
        for k, v in meta_lines:
            st.markdown(f"**{k}:** {v}")

st.markdown("---")
st.markdown(
    f"<div style='padding:14px;border-radius:12px;border:1px solid #ddd;"
    f"background:#fafafa;white-space:pre-wrap;line-height:1.5'>{text}</div>",
    unsafe_allow_html=True,
)
st.markdown("---")

# Widget keys
suf = slugify_col(model_col)
note_key = f"note_{row_id}_{suf}"
sel_key = f"sel_{row_id}_{suf}"

# init widget state once per item
if note_key not in st.session_state:
    st.session_state[note_key] = existing_note
if sel_key not in st.session_state:
    st.session_state[sel_key] = existing_rating  # can be ""

st.text_area("Note (optional)", key=note_key, height=90)

st.subheader("Select rating (A–D)")
st.radio(
    "Choose one:",
    options=[""] + RATINGS,
    format_func=lambda x: "— (not selected)" if x == "" else f"{x} — {RATING_LABELS[x]}",
    key=sel_key,
    horizontal=True,
)
st.caption(f"Current selection: **{st.session_state[sel_key] or '-'}**")

# -----------------------
# ACTIONS
# -----------------------
action_cols = st.columns(3)

with action_cols[0]:
    if st.button("Prev", use_container_width=True):
        st.session_state.pos = clamp_pos(st.session_state.pos - 1, len(pool))
        st.rerun()

with action_cols[1]:
    if st.button("Save", use_container_width=True):
        # Save even if rating empty (allows clearing)
        write_rating(df, row_id, model_col, st.session_state[sel_key], st.session_state[note_key])
        st.success("Saved.")

with action_cols[2]:
    if st.button("Next", use_container_width=True):
        # Save first (like autosave-on-next), but DO NOT force a rating.
        write_rating(df, row_id, model_col, st.session_state[sel_key], st.session_state[note_key])
        st.session_state.pos = clamp_pos(st.session_state.pos + 1, len(pool))
        st.rerun()

st.markdown("")

# Quick utilities
util_cols = st.columns(2)
with util_cols[0]:
    if st.button("Clear rating for this item", use_container_width=True):
        st.session_state[sel_key] = ""
        write_rating(df, row_id, model_col, "", st.session_state[note_key])
        st.success("Cleared rating.")
with util_cols[1]:
    if st.button("Clear note for this item", use_container_width=True):
        st.session_state[note_key] = ""
        write_rating(df, row_id, model_col, st.session_state[sel_key], "")
        st.success("Cleared note.")

# -----------------------
# EXPORT
# -----------------------
st.subheader("Export updated CSV")

default_name = f"{uploaded_file.name.replace('.csv','')}_rated_{rater_name.strip().replace(' ','_')}.csv"
export_name = st.text_input("Download filename", value=default_name)

csv_bytes = df.to_csv(index=False, sep=sep_used, quoting=csv.QUOTE_MINIMAL).encode("utf-8")
st.download_button(
    label="Download updated CSV",
    data=csv_bytes,
    file_name=export_name,
    mime="text/csv",
)

st.info("Tip: At the end, download the updated CSV and send it back to Neda.")
