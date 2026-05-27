import streamlit as st
import os

from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)

from langchain_chroma import Chroma

# -------------------------
# LOAD ENV VARIABLES
# -------------------------

load_dotenv(dotenv_path=".env")

# -------------------------
# PAGE TITLE
# -------------------------

st.set_page_config(page_title="Spotify RAG App")

st.title("🎵 Spotify RAG App")

st.write(
    "Ask questions about songs in the Spotify dataset."
)

# -------------------------
# EMBEDDINGS
# -------------------------

from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004",
google_api_key="AIzaSyCFj-UUEIe1K6qWZmx-O7PscVeO6ahouCg"
)


# -------------------------
# LOAD VECTOR DATABASE
# -------------------------

vectorstore = Chroma(
    persist_directory="chroma_spotify_db",
    embedding_function=embeddings
)

# -------------------------
# RETRIEVER
# -------------------------

retriever = vectorstore.as_retriever()

# -------------------------
# LLM
# -------------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key="AIzaSyCFj-UUEIe1K6qWZmx-O7PscVeO6ahouCg",
    temperature=0.7
)

# -------------------------
# USER INPUT
# -------------------------

question = st.text_input(
    "Ask a question about songs:"
)

# -------------------------
# RAG PIPELINE
# -------------------------

if question:

    # Retrieve documents
    docs = retriever.invoke(question)

    # -------------------------
    # WITHOUT LLM
    # -------------------------

    st.subheader("🔎 Without LLM")

    st.write(
        "These are the raw retrieved documents from the vector database."
    )

    for doc in docs:
        st.write(doc.page_content)
        st.write("---")

    # -------------------------
    # CREATE CONTEXT
    # -------------------------

    context = "\n".join(
        [doc.page_content for doc in docs]
    )

    # -------------------------
    # PROMPT
    # -------------------------

    prompt = f"""
    You are a music recommendation assistant.

    Answer the question ONLY using the context below.

    Context:
    {context}

    Question:
    {question}
    """

    # -------------------------
    # WITH LLM
    # -------------------------

    response = llm.invoke(prompt)

    st.subheader("🤖 With LLM")

    st.write(response.content)
