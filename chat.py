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
extra_stop = {"yang", "dan", "di", "ke", "dari", "pada", "untuk",
              "adalah", "dengan", "atau", "sebagai", "apa", "saya"}
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
    
    # gunakan stemmer indonesia (lebih akurat)
    try:
        tokens_stem = [stemmer.stem(t) for t in tokens_clean]
    except:
        tokens_stem = tokens_clean

    clean = " ".join(tokens_stem)

    return clean


# =========================
#  LOAD DATASET FIX
# =========================
DATA_PATH = "translated_computer_science_dataset.csv"

@st.cache_data
def load_dataset(path=DATA_PATH):
    df = pd.read_csv(path, engine="python", on_bad_lines="skip")

    # Perbaiki header yang rusak ;;;;;;;;;;;;;;;;;;
    new_cols = []
    for c in df.columns:
        c = c.strip().replace("\ufeff", "")
        c = c.split(";")[0]
        new_cols.append(c)
    df.columns = new_cols

    # Bersihkan jawaban dari ;;;;;;
    df = df.replace({";+": ""}, regex=True)

    if "input_id" not in df.columns or "output_id" not in df.columns:
        st.error(f"Dataset rusak, kolom ditemukan: {df.columns}")
        st.stop()

    return df[["input_id", "output_id"]].dropna().reset_index(drop=True)


# =========================
#       CHATBOT
# =========================
class Chatbot:
    def __init__(self, df, threshold=0.18):
        self.df = df.copy()
        self.df["clean_input"] = self.df["input_id"].apply(clean_text)

        # gunakan ngram lebih besar agar akurat seperti lokal
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 3), min_df=1)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["clean_input"])
        self.threshold = threshold

    def get_response(self, user_text):
        clean_user = clean_text(user_text)
        vec = self.vectorizer.transform([clean_user])
        sims = cosine_similarity(vec, self.tfidf_matrix).flatten()

        # penalti jarak berbasis panjang jawaban
        sims = sims * (1 - np.abs(np.log1p(self.df["clean_input"].str.len() - len(clean_user))))

        best_idx = np.argmax(sims)
        best_score = sims[best_idx]

        if best_score < self.threshold:
            return None, "❌ Tidak ada jawaban relevan.", best_score

        q = self.df.loc[best_idx, "input_id"]
        a = self.df.loc[best_idx, "output_id"]

        # bersihkan jawaban dari ;;;;;;;;
        a = re.sub(r";+", "", str(a)).strip()

        return q, a, best_score


# =========================
#       UI STREAMLIT
# =========================
st.set_page_config(page_title="Chatbot Ilmu Komputer", page_icon="🤖")

st.title("💬 Chatbot Pembelajaran Ilmu Komputer")
st.caption("Versi Peningkatan Akurasi • TF-IDF + N-gram 1–3 + Sastrawi Stemmer")
st.caption("Kelompok 11")

df = load_dataset()
bot = Chatbot(df)

st.divider()

user_input = st.text_area("Masukkan pertanyaan:", height=100)

if st.button("💬 Kirim"):
    if not user_input.strip():
        st.warning("Isi pertanyaan dulu.")
    else:
        with st.spinner("Sedang mencari jawaban..."):
            best_q, best_a, score = (
                bot.get_response(user_input)
                if bot.get_response(user_input)[0]
                else (None, bot.get_response(user_input)[1], bot.get_response(user_input)[2])
            )

        st.subheader("📌 Hasil")

        if best_q is None:
            st.error(best_a)
            st.write(f"📉 Similarity: `{score:.5f}`")
        else:
            st.success("Jawaban ditemukan!")
            st.write(f"**🔎 Pertanyaan mirip:** {best_q}")
            st.write(f"**💬 Jawaban:** {best_a}")
            st.write(f"📈 Similarity: `{score:.5f}`")

st.divider()
st.caption("© 2025 Chatbot Ilmu Komputer — Kelompok 11")
