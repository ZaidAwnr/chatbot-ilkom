import os
import re
import numpy as np
import pandas as pd
import nltk
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- NLTK Setup ---
for pkg in ["punkt", "stopwords"]:
    nltk.download(pkg, quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# ===========================================
#     ⬇ BAGIAN DATASET (DARI KODE KEDUA)
# ===========================================

DATA_PATH = r"DATA_PATH = "translated_computer_science_dataset.csv""

@st.cache_data
def load_dataset(path=DATA_PATH):
    df = pd.read_csv(path)
    return df[["input_id", "output_id"]].dropna().reset_index(drop=True)

# ===========================================
#     ⬆ HANYA BAGIAN INI YANG DIGANTI
# ===========================================


# --- Stopword Bahasa Indonesia (kode pertama) ---
stop_id = set(stopwords.words("indonesian"))
extra_stop = {"yang", "dan", "di", "ke", "dari", "pada", "untuk", "adalah", "dengan", "atau", "sebagai"}
stop_id = stop_id.union(extra_stop)

# --- Stemmer (kode pertama) ---
stemmer = PorterStemmer()

# --- Preprocessing (kode pertama) ---
def clean_text(text):
    if not isinstance(text, str):
        text = str(text)

    text = text.lower()
    text = re.sub(r"http\S+|www\S+|\S+@\S+", " ", text)
    text = re.sub(r"[^0-9a-zA-Z\u00C0-\u024F\u1E00-\u1EFF\s]", " ", text)

    tokens = word_tokenize(text)
    tokens_clean = [t for t in tokens if t not in stop_id and len(t) > 1]
    tokens_stem = [stemmer.stem(t) for t in tokens_clean]

    clean = " ".join(tokens_stem)
    return clean, tokens, tokens_clean, tokens_stem, clean

# --- Chatbot Class (kode pertama) ---
class Chatbot:
    def __init__(self, df, threshold=0.25):
        self.df = df.copy()
        self.df["clean_input"] = self.df["input_id"].apply(lambda x: clean_text(x)[4])

        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["clean_input"])
        self.threshold = threshold

    def get_response(self, user_text):
        _, tokens, tokens_clean, tokens_stem, clean_user = clean_text(user_text)

        vec = self.vectorizer.transform([clean_user])
        sims = cosine_similarity(vec, self.tfidf_matrix).flatten()

        best_idx = np.argmax(sims)
        best_score = sims[best_idx]

        # Jika pertanyaan di luar konteks
        if best_score <= 0.007:
            return None, "⚠️ Pertanyaan Anda di luar konteks pembelajaran ilmu komputer.", best_score

        # Jika similarity rendah
        if best_score < self.threshold:
            return None, "❌ Maaf, tidak ditemukan jawaban relevan.", best_score

        best_q = self.df.loc[best_idx, "input_id"]
        best_a = self.df.loc[best_idx, "output_id"]
        return best_q, best_a, best_score


# =============================
#       STREAMLIT UI
# =============================
st.set_page_config(page_title="Chatbot Ilmu Komputer", page_icon="🤖")

st.title("💬 Chatbot Pembelajaran Ilmu Komputer")
st.caption("NLTK + TF-IDF + Cosine Similarity • Dataset versi kode kedua")
st.caption("Kelompok 11")

df = load_dataset()
bot = Chatbot(df, threshold=0.25)

st.divider()
user_input = st.text_area("Ketik pertanyaan Anda:", height=100)

if st.button("💬 Kirim Pertanyaan"):
    if not user_input.strip():
        st.warning("Masukkan pertanyaan terlebih dahulu.")
    else:
        with st.spinner("Mencari jawaban terbaik..."):
            result = bot.get_response(user_input)

        best_q, best_a, score = result if result[0] else (None, result[1], result[2])

        st.subheader("📌 Hasil")

        # Jika tidak relevan
        if best_q is None:
            st.error(best_a)
            st.markdown(f"📉 **Similarity:** `{score:.5f}`")
        else:
            st.success("Ditemukan pertanyaan paling mirip!")
            st.markdown(f"**🔎 Pertanyaan mirip:** {best_q}")
            st.markdown(f"**💬 Jawaban:** {best_a}")
            st.markdown(f"📈 **Similarity:** `{score:.5f}`")

st.divider()
st.caption("© 2025 Chatbot Ilmu Komputer | Kelompok 11")

