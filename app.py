import os
import streamlit as st
from llama_cpp import Llama
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

# ============ SETUP LOCAL LLM ============

# Use absolute path for reliability
MODEL_PATH = r"C:\Users\kedha\multi-agent-rag-chat-bot\models\mistral-7b-instruct-v0.2.Q4_K_M.gguf"

# Load LLaMA/Mistral model with optimized parameters for low memory usage
llm = Llama(model_path=MODEL_PATH, n_ctx=2048, n_batch=256)

# ============ SETUP CHROMADB =============

# Load lightweight embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Define persistent storage directory for vector database
CHROMA_DB_DIR = "chroma_db"
vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)

# ============ STREAMLIT UI ==============
st.title("🔍 Local Multi-Agent RAG Chatbot")

query = st.text_input("💬 Enter your query:")

if query:
    with st.spinner("🔎 Retrieving documents..."):
        # Retrieve relevant documents
        retrieved_docs = vectorstore.similarity_search(query, k=3)
        doc_texts = " ".join([doc.page_content for doc in retrieved_docs])

    with st.spinner("🤖 Generating response..."):
        # Generate response using local LLM
        prompt = f"Here are some relevant documents: {doc_texts}\nAnswer the following query: {query}"
        response = llm(prompt)

    # Display response
    st.subheader("AI Response:")
    st.write(response["choices"][0]["text"])
