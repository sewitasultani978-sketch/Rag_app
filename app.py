import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import os

from langchain_core.documents import Document
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)
from langchain_community.vectorstores import Chroma

# =========================
# Load API key
# =========================

load_dotenv(".env")

# =========================
# Spotify Theme UI
# =========================

st.set_page_config(
    page_title="Spotify RAG App",
    page_icon="🎵",
    layout="centered"
)

st.markdown("""
<style>

/* Background */
.stApp {
    background:
        linear-gradient(rgba(0,0,0,0.82), rgba(0,0,0,0.92)),
        url("https://images.unsplash.com/photo-1493225457124-a3eb161ffa5f?q=80&w=2070&auto=format&fit=crop");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    color: white;
}

/* Main card */
.block-container {
    background-color: rgba(18,18,18,0.78);
    padding: 3rem;
    border-radius: 25px;
    backdrop-filter: blur(12px);
    box-shadow: 0 0 25px rgba(0,0,0,0.5);
    margin-top: 40px;
}

/* Titles */
h1, h2, h3 {
    color: white;
    text-align: center;
}

/* Subtitle */
.subtitle {
    text-align: center;
    color: #b3b3b3;
    font-size: 20px;
    margin-bottom: 30px;
}

/* Divider */
hr {
    border: 1px solid #282828;
}

/* Input field */
.stTextInput > div > div > input {
    background-color: #282828;
    color: white;
    border-radius: 12px;
    border: 2px solid #1DB954;
    padding: 14px;
    font-size: 16px;
}

/* Input label */
label {
    color: white !important;
    font-weight: bold;
}

/* Button */
.stButton > button {
    background: linear-gradient(90deg, #1DB954, #1ed760);
    color: white;
    border-radius: 50px;
    border: none;
    padding: 12px 30px;
    font-weight: bold;
    font-size: 16px;
    transition: 0.3s;
}

/* Button hover */
.stButton > button:hover {
    transform: scale(1.05);
    box-shadow: 0 0 20px #1DB954;
    color: black;
}

/* Response box */
.response-box {
    background-color: rgba(24,24,24,0.95);
    padding: 25px;
    border-radius: 18px;
    border-left: 6px solid #1DB954;
    margin-top: 30px;
    box-shadow: 0 0 20px rgba(0,0,0,0.4);
}

/* Spinner */
.stSpinner > div {
    color: #1DB954;
}

</style>
""", unsafe_allow_html=True)

# =========================
# Title
# =========================

st.markdown("# 🎵 Spotify RAG Application")

st.markdown(
    '<p class="subtitle">Fråga AI:n om musik baserat på Spotify-data 🎧</p>',
    unsafe_allow_html=True
)

st.markdown("---")

# =========================
# Read dataset
# =========================
df = pd.read_csv("Data/cleaned_dataset.csv")

# =========================
# Create embeddings model
# =========================

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

# =========================
# Create combined text
# =========================

df["text"] = (
    "Track name: " + df["track_name"].astype(str)
    + ". Artist: " + df["artists"].astype(str)
    + ". Album: " + df["album_name"].astype(str)
    + ". Genre: " + df["track_genre"].astype(str)
    + ". Popularity: " + df["popularity"].astype(str)
)

# =========================
# Create documents
# =========================

documents = []

for text in df["text"][:10]:
    documents.append(Document(page_content=text))

# =========================
# Create vector database
# =========================

vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    collection_name="spotify"
)

retriever = vectorstore.as_retriever()

# =========================
# Load Gemini model
# =========================

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.3
)

# =========================
# User input
# =========================

question = st.text_input(
    "🔍 Skriv en fråga om musik:"
)

# =========================
# Generate answer
# =========================

if st.button("🎧 Generera svar"):

    if question:

        try:

            with st.spinner("AI:n tänker..."):

                docs = retriever.invoke(question)[:1]

                context = docs[0].page_content

                final_prompt = f"""
                Du är en musikexpert.

                Context:
                {context}

                Question:
                {question}
                """

                response = llm.invoke(final_prompt)

                st.markdown(
                    f"""
                    <div class="response-box">
                        <h3>🎶 Svar</h3>
                        <p>{response.content}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        except Exception as e:
            st.error(f"Fel: {e}")