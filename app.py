import os
import streamlit as st
from llama_cpp import Llama
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

# ============ SETUP LOCAL LLM ============

# Use absolute path for reliability
MODEL_PATH = r"C:\Users\kedha\multi-agent-rag-chat-bot\models\mistral-7b-instruct-v0.2.Q4_K_M.gguf"

# Load LLaMA/Mistral model with optimized parameters for low memory usage
llm = Llama(model_path=MODEL_PATH, n_ctx=2048, n_batch=128, cache=True)  # Reduced context and batch for speed

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
        doc_texts = " ".join([doc.page_content for doc in retrieved_docs]) if retrieved_docs else "No relevant documents found."

    with st.spinner("🤖 Generating response..."):
        # Generate response using local LLM with streaming
        prompt = f"You are a helpful AI assistant. Answer the question in a conversational way.\nContext: {doc_texts}\nUser: {query}\nAI:"
        response = llm(prompt, max_tokens=256, stream=True)  # Enabled streaming for faster response
        full_response = "".join(chunk["choices"][0]["text"] for chunk in response).strip() if "choices" in response else "I'm sorry, I couldn't generate a response."

    # Display response
    st.subheader("AI Response:")
    st.write(full_response)

    # Debugging: Print full response to console
    print(full_response)
