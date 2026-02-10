# daily2.py
import random
import csv
import math
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

ITEMS_PER_BATCH_DEFAULT = 40  # همون چیزی که الان داری

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

def robust_read_csv(file_like):
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

def ensure_model_columns_exist(df: pd.DataFrame):
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

def build_item_pool(df: pd.DataFrame):
    pool = []
    for model_col in MODEL_TEXT_COLS:
        for row_id, val in df[model_col].items():
            t = clean_cell(val)
            if t:
                pool.append({"row_id": row_id, "model_col": model_col, "text": t})
    if not pool:
        st.error("No non-empty dialogs found across the specified model columns.")
        st.stop()
    return pool

def write_rating_to_df(df: pd.DataFrame, row_id: int, model_col: str, rating: str, note: str):
    suf = slugify_col(model_col)
    rating_col = f"{RATING_PREFIX}{suf}"
    note_col = f"{NOTE_PREFIX}{suf}"
    df.at[row_id, rating_col] = clean_cell(rating)
    df.at[row_id, note_col] = clean_cell(note)

def clamp_pos(n_items: int):
    if n_items <= 0:
        st.session_state.pos = 0
    else:
        st.session_state.pos = max(0, min(st.session_state.pos, n_items - 1))

def get_qp_int(key: str, default: int) -> int:
    try:
        v = st.query_params.get(key)
        if v is None:
            return default
        if isinstance(v, list):
            v = v[0]
        return int(v)
    except Exception:
        return default

def set_qp_int(key: str, value: int):
    st.query_params[key] = str(int(value))

# =========================
# SIDEBAR
# =========================
st.sidebar.title("Setup")
rater_name = st.sidebar.text_input("Rater name (required)", value="")
uploaded_file = st.sidebar.file_uploader("Upload your CSV", type=["csv"])

st.sidebar.divider()
items_per_batch = st.sidebar.number_input(
    "Items per batch",
    min_value=10,
    max_value=500,
    value=ITEMS_PER_BATCH_DEFAULT,
    step=10,
)

if not rater_name.strip():
    st.info("Enter your name in the sidebar to start.")
    st.stop()

if uploaded_file is None:
    st.info("Upload your CSV in the sidebar to start.")
    st.stop()

# =========================
# LOAD DF + POOL ONCE PER FILE
# =========================
file_signature = f"{uploaded_file.name}-{uploaded_file.size}"

if st.session_state.get("file_signature") != file_signature:
    df0, sep0 = robust_read_csv(uploaded_file)
    df0 = ensure_model_columns_exist(df0)

    st.session_state.file_signature = file_signature
    st.session_state.source_label = f"Uploaded: {uploaded_file.name}"
    st.session_state.sep_used = sep0
    st.session_state.df = df0
    st.session_state.item_pool = build_item_pool(st.session_state.df)

df = st.session_state.df
sep_used = st.session_state.sep_used
source_label = st.session_state.source_label
pool = st.session_state.item_pool

total_batches = max(1, math.ceil(len(pool) / int(items_per_batch)))

# batch from URL (persist across refresh)
batch_from_url = get_qp_int("batch", 1)
batch_from_url = max(1, min(batch_from_url, total_batches))

st.sidebar.subheader("Batch")
batch_num = st.sidebar.selectbox(
    "Select batch",
    options=list(range(1, total_batches + 1)),
    index=batch_from_url - 1,
)

# keep URL in sync
if batch_num != batch_from_url:
    set_qp_int("batch", batch_num)
    # reset position/order when batch changes
    st.session_state.pos = 0
    st.session_state.order = None
    st.rerun()

# define slice for this batch
start = (batch_num - 1) * int(items_per_batch)
end = min(start + int(items_per_batch), len(pool))
batch_items = pool[start:end]

if not batch_items:
    st.error("This batch is empty. Try a different batch number.")
    st.stop()

# init order/pos per (file + batch + batch_size)
batch_signature = f"{file_signature}-batch{batch_num}-k{items_per_batch}"
if st.session_state.get("batch_signature") != batch_signature:
    st.session_state.batch_signature = batch_signature
    st.session_state.order = list(range(len(batch_items)))
    random.shuffle(st.session_state.order)
    st.session_state.pos = 0

st.sidebar.subheader("Controls")
if st.sidebar.button("Reshuffle order (this batch)"):
    st.session_state.order = list(range(len(batch_items)))
    random.shuffle(st.session_state.order)
    st.session_state.pos = 0
    st.rerun()

if st.sidebar.button("Next batch →"):
    if batch_num < total_batches:
        set_qp_int("batch", batch_num + 1)
        st.session_state.pos = 0
        st.session_state.order = None
        st.rerun()
    else:
        st.sidebar.info("You are already on the last batch.")

# =========================
# MAIN UI
# =========================
st.title("Culture Rater")
st.caption(" • ".join([source_label, f"Total items: {len(pool)}", f"Rater: {rater_name}"]))
st.info(f"Batch: **{batch_num} / {total_batches}** • Items in this batch: **{len(batch_items)}** (slice {start+1}-{end})")

with st.expander("A/B/C/D meaning (confidence it is my country)", expanded=False):
    st.markdown("- **A** — Very sure: Unmistakably my country’s culture.")
    st.markdown("- **B** — Fairly sure: Mostly fits my country; a few details could fit elsewhere.")
    st.markdown("- **C** — Not sure: Could be many countries; weak or generic cultural cues.")
    st.markdown("- **D** — Sure it is NOT mine: Clearly mismatched; points elsewhere or feels wrong.")

if "pos" not in st.session_state:
    st.session_state.pos = 0
clamp_pos(len(batch_items))

idx_in_batch = st.session_state.order[st.session_state.pos]
item = batch_items[idx_in_batch]

row_id = item["row_id"]
model_col = item["model_col"]
text = item["text"]

suf = slugify_col(model_col)
rating_col = f"{RATING_PREFIX}{suf}"
note_col = f"{NOTE_PREFIX}{suf}"

existing_note = clean_cell(df.at[row_id, note_col])
existing_rating = clean_cell(df.at[row_id, rating_col])

st.write(f"Item {st.session_state.pos + 1} / {len(batch_items)} (Batch {batch_num}/{total_batches})")
st.caption(f"Model column: **{model_col}**  •  Saved rating: **{existing_rating or '-'}**")

st.markdown("---")
st.markdown(
    f"<div style='padding:14px;border-radius:10px;border:1px solid #ddd;background:#fafafa;white-space:pre-wrap;'>{text}</div>",
    unsafe_allow_html=True,
)
st.markdown("---")

note_key = f"note_{batch_num}_{row_id}_{suf}"
sel_key = f"selected_{batch_num}_{row_id}_{suf}"

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

# =========================
# NAV
# =========================
nav_cols = st.columns(2)

with nav_cols[0]:
    if st.button("Prev", use_container_width=True):
        st.session_state.pos -= 1
        clamp_pos(len(batch_items))
        st.rerun()

with nav_cols[1]:
    if st.button("Next (save & go)", use_container_width=True):
        if not st.session_state[sel_key]:
            st.warning("Please select A/B/C/D before going to next.")
            st.stop()

        write_rating_to_df(
            st.session_state.df,
            row_id,
            model_col,
            st.session_state[sel_key],
            st.session_state[note_key],
        )

        if st.session_state.pos < len(batch_items) - 1:
            st.session_state.pos += 1
        clamp_pos(len(batch_items))
        st.rerun()

# =========================
# EXPORT
# =========================
st.subheader("Export updated CSV (this batch only)")
default_name = f"{uploaded_file.name.replace('.csv','')}_batch_{batch_num:02d}_rated_{rater_name.strip().replace(' ','_')}.csv"
export_name = st.text_input("Download filename", value=default_name)

csv_bytes = st.session_state.df.to_csv(index=False, sep=sep_used, quoting=csv.QUOTE_MINIMAL).encode("utf-8")
st.download_button(
    label="Download updated CSV",
    data=csv_bytes,
    file_name=export_name,
    mime="text/csv",
)

st.caption("After finishing this batch, download the CSV and upload it using your submission form.")
