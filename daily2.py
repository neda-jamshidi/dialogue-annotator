import os
import random
import csv
import argparse

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Culture Rater", layout="centered")

# =========================
# CONFIG
# =========================
MODEL_TEXT_COLS = [
    "Chatgpt 5.2",
    "Gemini Pro",
    "Qwen 3-Max",
    "DeepSeek v3.2",
    "Mistral Large",
]

RATING_PREFIX = "confidence_rating_"   # e.g., confidence_rating_Chatgpt_5_2
NOTE_PREFIX = "rating_note_"           # e.g., rating_note_Chatgpt_5_2

# =========================
# CLI ARGS (reliable for Streamlit)
# Run:
#   cd ~/Desktop
#   streamlit run culture_rater.py -- --csv "/Users/<you>/Desktop/merged_dialogues.csv.csv" --rater "Neda"
# =========================
def get_cli_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--csv", required=True, help="Path to input CSV")
    parser.add_argument("--rater", default="", help="Annotator name/id (shown, not saved)")
    args, _ = parser.parse_known_args()
    return args

ARGS = get_cli_args()
CSV_PATH = ARGS.csv
RATER_NAME = ARGS.rater

# =========================
# Helpers
# =========================
def clean_cell(x) -> str:
    """Return empty string for NaN/None/'nan', else stripped string."""
    if pd.isna(x):
        return ""
    s = str(x).strip()
    return "" if s.lower() == "nan" else s

def slugify_col(col_name: str) -> str:
    """Make a safe suffix for column names."""
    s = "".join(ch if ch.isalnum() else "_" for ch in col_name.strip())
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")

# =========================
# ROBUST CSV READER
# =========================
def robust_read_csv(path_or_file):
    for sep in [",", ";", "\t", "|"]:
        try:
            df = pd.read_csv(
                path_or_file,
                sep=sep,
                engine="python",
                quoting=csv.QUOTE_MINIMAL,
            )
            if len(df.columns) == 1 and sep != "|":
                continue
            return df, sep
        except Exception:
            pass

    # last resort
    try:
        df = pd.read_csv(
            path_or_file,
            sep=",",
            engine="python",
            on_bad_lines="skip",
            quoting=csv.QUOTE_MINIMAL,
        )
        return df, ","
    except Exception as e:
        st.error(f"Could not read CSV.\n\nError: {e}")
        st.stop()

def ensure_model_columns_exist(df: pd.DataFrame):
    # check model text columns exist
    missing = [c for c in MODEL_TEXT_COLS if c not in df.columns]
    if missing:
        st.error(
            "Some required text columns are missing.\n\n"
            f"Missing: {missing}\n\n"
            f"Available columns: {list(df.columns)}"
        )
        st.stop()

    # add rating + note columns per model
    for c in MODEL_TEXT_COLS:
        suf = slugify_col(c)
        rating_col = f"{RATING_PREFIX}{suf}"
        note_col = f"{NOTE_PREFIX}{suf}"
        if rating_col not in df.columns:
            df[rating_col] = ""
        if note_col not in df.columns:
            df[note_col] = ""
    return df

def load_data(csv_path: str):
    if not csv_path:
        st.error("Missing --csv argument.")
        st.stop()

    if not os.path.exists(csv_path):
        st.error(f"File not found: {csv_path}")
        st.stop()

    df, sep = robust_read_csv(csv_path)
    df = ensure_model_columns_exist(df)
    return df, sep, f"Path: {csv_path}", csv_path

def build_items(df: pd.DataFrame):
    """
    Build ONE mixed pool of items from ALL model columns.
    Each item knows: row_id, model_col, text
    """
    items = []
    for model_col in MODEL_TEXT_COLS:
        for row_id, val in df[model_col].items():
            t = clean_cell(val)
            if t:
                items.append({"row_id": row_id, "model_col": model_col, "text": t})

    if not items:
        st.error("No non-empty dialogs found across the specified model columns.")
        st.stop()

    return items

def init_state(n_items: int):
    # shuffle ONCE per session; stable across reruns
    if "order" not in st.session_state or len(st.session_state.order) != n_items:
        st.session_state.order = list(range(n_items))
        random.shuffle(st.session_state.order)
        st.session_state.pos = 0

def clamp_pos(n_items: int):
    st.session_state.pos = max(0, min(st.session_state.pos, n_items - 1))

def current_item_index():
    return st.session_state.order[st.session_state.pos]

def write_rating_to_df(df: pd.DataFrame, row_id: int, model_col: str, rating: str, note: str):
    suf = slugify_col(model_col)
    rating_col = f"{RATING_PREFIX}{suf}"
    note_col = f"{NOTE_PREFIX}{suf}"
    df.at[row_id, rating_col] = clean_cell(rating)
    df.at[row_id, note_col] = clean_cell(note)

def persist_df_to_disk(df: pd.DataFrame, writeback_path: str, sep_used: str) -> bool:
    if not writeback_path:
        return False

    try:
        # backup original first
        backup_path = writeback_path + ".bak"
        try:
            if os.path.exists(writeback_path):
                with open(writeback_path, "rb") as src, open(backup_path, "wb") as dst:
                    dst.write(src.read())
        except Exception:
            pass

        df.to_csv(writeback_path, index=False, sep=sep_used, quoting=csv.QUOTE_MINIMAL)
        return True
    except Exception as e:
        st.error(f"Could not save back to CSV: {e}")
        return False

# =========================
# LOAD + PREPARE
# =========================
df, sep_used, source_label, writeback_path = load_data(CSV_PATH)
items = build_items(df)

init_state(len(items))
clamp_pos(len(items))

# =========================
# SIDEBAR (only reshuffle)
# =========================
st.sidebar.subheader("Controls")
if st.sidebar.button("Reshuffle order"):
    st.session_state.order = list(range(len(items)))
    random.shuffle(st.session_state.order)
    st.session_state.pos = 0
    st.rerun()

# =========================
# MAIN UI
# =========================
st.title("Culture Rater")

caption_bits = [source_label, f"Items: {len(items)}"]
if RATER_NAME:
    caption_bits.append(f"Rater: {RATER_NAME} (not saved)")
st.caption(" • ".join(caption_bits))

with st.expander("A/B/C/D meaning (confidence it is my country)", expanded=True):
    st.markdown("- **A** — Very sure: Unmistakably my country’s culture.")
    st.markdown("- **B** — Fairly sure: Mostly fits my country; a few details could fit elsewhere.")
    st.markdown("- **C** — Not sure: Could be many countries; weak or generic cultural cues.")
    st.markdown("- **D** — Sure it is NOT mine: Clearly mismatched; points elsewhere or feels wrong.")

idx_in_items = current_item_index()
item = items[idx_in_items]
row_id = item["row_id"]
model_col = item["model_col"]
text = item["text"]

suf = slugify_col(model_col)
rating_col = f"{RATING_PREFIX}{suf}"
note_col = f"{NOTE_PREFIX}{suf}"

existing_note = clean_cell(df.at[row_id, note_col])
existing_rating = clean_cell(df.at[row_id, rating_col])

st.write(f"Item {st.session_state.pos + 1} / {len(items)}")
st.caption(f"Model column: **{model_col}**  •  Existing rating: **{existing_rating or '-'}**")

st.markdown("---")
st.markdown(
    f"<div style='padding:14px;border-radius:10px;border:1px solid #ddd;background:#fafafa;white-space:pre-wrap;'>{text}</div>",
    unsafe_allow_html=True,
)
st.markdown("---")

note_key = f"note_{row_id}_{suf}"
if note_key not in st.session_state:
    st.session_state[note_key] = existing_note

st.text_area("Note (optional)", key=note_key, height=90)

# =========================
# ONLY: A/B/C/D + Prev/Next
# =========================
st.subheader("Rate (A–D)")
btn_cols = st.columns(4)
rating_clicked = None
for col, label in zip(btn_cols, ["A", "B", "C", "D"]):
    with col:
        if st.button(label, use_container_width=True, key=f"rate_{label}_{row_id}_{suf}_{st.session_state.pos}"):
            rating_clicked = label

# Click A/B/C/D -> save immediately + go next
if rating_clicked:
    write_rating_to_df(df, row_id, model_col, rating_clicked, st.session_state[note_key])
    persist_df_to_disk(df, writeback_path, sep_used)

    if st.session_state.pos < len(items) - 1:
        st.session_state.pos += 1
    clamp_pos(len(items))
    st.rerun()

nav_cols = st.columns(2)
with nav_cols[0]:
    if st.button("Prev", use_container_width=True):
        st.session_state.pos -= 1
        clamp_pos(len(items))
        st.rerun()

with nav_cols[1]:
    if st.button("Next", use_container_width=True):
        st.session_state.pos += 1
        clamp_pos(len(items))
        st.rerun()

# =========================
# EXPORT
# =========================
st.subheader("Export updated CSV")
export_name = st.text_input("Download filename", value="merged_dialogues.csv_with_ratings.csv")

csv_bytes = df.to_csv(index=False, sep=sep_used, quoting=csv.QUOTE_MINIMAL).encode("utf-8")
st.download_button(
    label="Download updated CSV",
    data=csv_bytes,
    file_name=export_name,
    mime="text/csv",
)

st.caption(f"Write-back is ON: {writeback_path} (backup: {writeback_path}.bak)")
