import random
import csv
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


# -------------------------
# helpers
# -------------------------
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
            except:
                pass
            df = pd.read_csv(file_like, sep=sep, engine="python")
            if len(df.columns) == 1 and sep != "|":
                continue
            return df, sep
        except:
            continue

    file_like.seek(0)
    df = pd.read_csv(file_like, sep=",", engine="python", on_bad_lines="skip")
    return df, ","


# -------------------------
# sidebar
# -------------------------
st.sidebar.title("Setup")

rater_name = st.sidebar.text_input("Rater name", "")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
batch_size = st.sidebar.number_input("Items per batch", 10, 200, 40)

if not rater_name:
    st.info("Enter rater name")
    st.stop()

if not uploaded_file:
    st.info("Upload CSV")
    st.stop()

# -------------------------
# load once
# -------------------------
file_sig = uploaded_file.name + str(uploaded_file.size)

if st.session_state.get("file_sig") != file_sig:
    df, sep = robust_read_csv(uploaded_file)

    for c in MODEL_TEXT_COLS:
        suf = slugify_col(c)
        if f"{RATING_PREFIX}{suf}" not in df.columns:
            df[f"{RATING_PREFIX}{suf}"] = ""
        if f"{NOTE_PREFIX}{suf}" not in df.columns:
            df[f"{NOTE_PREFIX}{suf}"] = ""

    pool = []
    for col in MODEL_TEXT_COLS:
        for rid, val in df[col].items():
            t = clean_cell(val)
            if t:
                pool.append((rid, col, t))

    st.session_state.file_sig = file_sig
    st.session_state.df = df
    st.session_state.sep = sep
    st.session_state.pool = pool

df = st.session_state.df
pool = st.session_state.pool
sep = st.session_state.sep

# -------------------------
# batch from URL (ORDERED)
# -------------------------
qp = st.query_params
batch_num = int(qp.get("batch", "1"))

total_batches = (len(pool) + batch_size - 1) // batch_size

batch_num = max(1, min(batch_num, total_batches))

start = (batch_num - 1) * batch_size
end = min(start + batch_size, len(pool))

batch_slice = pool[start:end]

# -------- shuffle ONLY inside batch --------
if "batch_shuffle_key" not in st.session_state or st.session_state.batch_shuffle_key != (file_sig, batch_num):
    order = list(range(len(batch_slice)))
    random.shuffle(order)
    st.session_state.batch_order = order
    st.session_state.batch_shuffle_key = (file_sig, batch_num)

order = st.session_state.batch_order
batch_items = [batch_slice[i] for i in order]

# -------------------------
# UI
# -------------------------
st.title("Culture Rater — Batch mode")
st.caption(f"Batch {batch_num}/{total_batches} • items {len(batch_items)}")

if "pos" not in st.session_state:
    st.session_state.pos = 0

st.session_state.pos = max(0, min(st.session_state.pos, len(batch_items)-1))

row_id, model_col, text = batch_items[st.session_state.pos]

suf = slugify_col(model_col)
rating_col = f"{RATING_PREFIX}{suf}"
note_col = f"{NOTE_PREFIX}{suf}"

st.write(f"Item {st.session_state.pos+1} / {len(batch_items)}")
st.markdown(text)

note_key = f"note_{row_id}_{suf}"
rate_key = f"rate_{row_id}_{suf}"

if note_key not in st.session_state:
    st.session_state[note_key] = clean_cell(df.at[row_id, note_col])

if rate_key not in st.session_state:
    st.session_state[rate_key] = clean_cell(df.at[row_id, rating_col])

st.text_area("Note", key=note_key)

st.radio("Rating", ["", "A","B","C","D"], key=rate_key, horizontal=True)

def save():
    df.at[row_id, rating_col] = st.session_state[rate_key]
    df.at[row_id, note_col] = st.session_state[note_key]

col1,col2 = st.columns(2)

with col1:
    if st.button("Prev"):
        save()
        st.session_state.pos -= 1
        st.rerun()

with col2:
    if st.button("Next"):
        save()
        st.session_state.pos += 1
        st.rerun()

# -------------------------
# export
# -------------------------
st.subheader("Download this batch result")

out = df.to_csv(index=False, sep=sep).encode("utf-8")

st.download_button(
    "Download CSV",
    out,
    f"rated_batch_{batch_num}_{rater_name}.csv"
)
