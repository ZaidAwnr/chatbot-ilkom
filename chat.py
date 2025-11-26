import os
import re
import numpy as np
import pandas as pd
import nltk
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================
#  DOWNLOAD STOPWORDS
# =========================
nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords

# =========================
#  STEMMER SASTRAWI
# =========================
try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    stemmer = StemmerFactory().create_stemmer()
except:
    from nltk.stem import PorterStemmer
    stemmer = PorterStemmer()


# =========================
#   STOPWORDS
# =========================
stop_id = set(stopwords.words("indonesian"))
extra_stop = {
    "yang", "dan", "di", "ke", "dari", "pada", "untuk",
    "adalah", "dengan", "atau", "sebagai", "apa", "saya",
    "itu", "ini", "karena", "jika"
}
stop_id |= extra_stop


# =========================
#  TOKENIZER MANUAL OPTIMAL
# =========================
def tokenize(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|\S+@\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split()


# =========================
#  PREPROCESSING
# =========================
def clean_text(text):
    if not isinstance(text, str):
        text = str(text)

    tokens = tokenize(text)
    tokens_clean = [t for t in tokens if t not in stop_id]

    try:
        tokens_stem = [stemmer.stem(t) for t in tokens_clean]
    except:
        tokens_stem = tokens_clean

    return " ".join(tokens_stem)


# =========================
#  LOAD DATASET SUPER FIX
# =========================
DATA_PATH = "translated_computer_science_dataset.csv"

@st.cache_data
def load_dataset(path=DATA_PATH):
    encodings = ["utf-8", "utf-8-sig", "latin1"]

    df = None
    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc, engine="python", on_bad_lines="skip")
            break
        except:
            continue

    if df is None:
        st.error("❌ Dataset gagal dibaca dengan semua encoding.")
        return None

    # ========== perbaiki header ==========
    cleaned_cols = []
    for c in df.columns:
        c = str(c).strip().replace("\ufeff", "")
        c = c.split(";")[0]  # hilangkan ;;;;
        cleaned_cols.append(c)

    df.columns = cleaned_cols

    # ========== bersihkan isi dataset ==========
    df = df.replace({";+": ""}, regex=True)

    # ========== validasi kolom ==========
    required = ["input_id", "output_id"]
    if any(col not in df.columns for col in required):
        st.error(f"""
        ❌ Dataset tidak memiliki kolom lengkap.

        Kolom ditemukan:
        {list(df.columns)}

        Kolom wajib:
        {required}
        """)
        return None

    df = df[["input_id", "output_id"]].dropna().reset_index(drop=True)
    return df


# =========================
#        CHATBOT
# =========================
class Chatbot:
    def __init__(self, df, threshold=0.20):
        self.df = df.copy()
        self.df["clean_input"] = self.df["input_id"].apply(clean_text)

        self.vectorizer = TfidfVectorizer(ngram_range=(1, 3), min_df=1)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["clean_input"])

        self.threshold = threshold

    def get_response(self, text):
        clean_user = clean_text(text)
        vec = self.vectorizer.transform([clean_user])
        sims = cosine_similarity(vec, self.tfidf_matrix).flatten()

        best_idx = np.argmax(sims)
        best_score = sims[best_idx]

        if best_score < self.threshold:
            return None, "❌ Tidak ada jawaban relevan.", best_score

        q = self.df.loc[best_idx, "input_id"]
        a = self.df.loc[best_idx, "output_id"]

        a = re.sub(r";+", "", str(a))

        return q, a, best_score


# =========================
#     STREAMLIT UI
# =========================
st.set_page_config(page_title="Chatbot Ilmu Komputer", page_icon="🤖")

st.title("💬 Chatbot Pembelajaran Ilmu Komputer")
st.caption("Versi Stabil • TF-IDF + N-gram 1–3 + Sastrawi Stemmer")
st.caption("Kelompok 11")

df = load_dataset()

if df is None:
    st.stop()  # hentikan UI jika dataset gagal

bot = Chatbot(df)

st.divider()

user_input = st.text_area("Masukkan pertanyaan Anda:", height=100)

if st.button("💬 Kirim"):
    if not user_input.strip():
        st.warning("Harap masukkan pertanyaan.")
    else:
        with st.spinner("Sedang mencari jawaban..."):
            result = bot.get_response(user_input)

        if result[0] is None:
            _, msg, score = result
            st.error(msg)
            st.write(f"📉 Similarity: `{score:.5f}`")
        else:
            q, a, score = result
            st.success("Jawaban ditemukan!")
            st.write(f"**🔎 Pertanyaan paling mirip:** {q}")
            st.write(f"**💬 Jawaban:** {a}")
            st.write(f"📈 Similarity: `{score:.5f}`")

st.divider()
st.caption("© 2025 Chatbot Ilmu Komputer — Kelompok 11")
