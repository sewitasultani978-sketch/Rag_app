import streamlit as st
import pandas as pd
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)
from langchain_community.vectorstores import Chroma


# Laddar API-nyckeln
load_dotenv(".env")


# Inställningar för sidan
st.set_page_config(
    page_title="Spotify RAG App",
    page_icon="🎵",
    layout="centered"
)


# Spotify-inspirerad design
st.markdown("""
<style>

.stApp {
    background:
        linear-gradient(rgba(0,0,0,0.82), rgba(0,0,0,0.92)),
        url("https://images.unsplash.com/photo-1493225457124-a3eb161ffa5f?q=80&w=2070&auto=format&fit=crop");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    color: white;
}

.block-container {
    background-color: rgba(18,18,18,0.78);
    padding: 3rem;
    border-radius: 25px;
    backdrop-filter: blur(12px);
    margin-top: 40px;
}

h1, h2, h3 {
    color: white;
    text-align: center;
}

.subtitle {
    text-align: center;
    color: #b3b3b3;
    font-size: 18px;
    margin-bottom: 30px;
}

.stTextInput > div > div > input {
    background-color: #282828;
    color: white;
    border-radius: 12px;
    border: 2px solid #1DB954;
    padding: 14px;
}

label {
    color: white !important;
    font-weight: bold;
}

.stButton > button {
    background: linear-gradient(90deg, #1DB954, #1ed760);
    color: white;
    border-radius: 50px;
    border: none;
    padding: 12px 28px;
    font-weight: bold;
    transition: 0.3s;
}

.stButton > button:hover {
    transform: scale(1.05);
    box-shadow: 0 0 18px #1DB954;
    color: black;
}

.response-box {
    background-color: rgba(24,24,24,0.95);
    padding: 25px;
    border-radius: 18px;
    border-left: 5px solid #1DB954;
    margin-top: 25px;
}

</style>
""", unsafe_allow_html=True)


# Titel
st.markdown("# 🎵 Spotify RAG Application")

st.markdown(
    '<p class="subtitle">Fråga AI:n om musik baserat på Spotify-data 🎧</p>',
    unsafe_allow_html=True
)

st.markdown("---")


# Läser in datasetet
df = pd.read_csv("Data/cleaned_dataset.csv")


# Embeddings-modell
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)


# Skapar text som används för sökning
df["text"] = (
    "Track: " + df["track_name"].astype(str)
    + ". Artist: " + df["artists"].astype(str)
    + ". Album: " + df["album_name"].astype(str)
    + ". Genre: " + df["track_genre"].astype(str)
    + ". Popularity: " + df["popularity"].astype(str)
)


# Gör om rader till dokument
documents = [
    Document(page_content=text)
    for text in df["text"][:10]
]


# Skapar vektordatabas
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    collection_name="spotify"
)

retriever = vectorstore.as_retriever()


# Gemini-modell
llm = ChatGoogleGenerativeAI(
    model="gemini-flash-latest",
    temperature=0.3
)


# Frågeruta
question = st.text_input("🔍 Skriv en fråga om musik:")


if st.button("🎧 Generera svar"):

    if question:

        try:

            with st.spinner("AI:n tänker..."):

                docs = retriever.invoke(question)[:1]

                context = docs[0].page_content

                prompt = f"""
                Du är en musikexpert.

                Context:
                {context}

                Question:
                {question}
                """

                response = llm.invoke(prompt)

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