import os
import streamlit as st
from llama_cpp import Llama
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

# ============ SETUP LOCAL LLM ============

MODEL_PATH = r"C:\Users\kedha\multi-agent-rag-chat-bot\models\mistral-7b-instruct-v0.2.Q4_K_M.gguf"

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_batch=128,
    n_threads=os.cpu_count(),
)

# ============ SETUP CHROMADB =============

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
CHROMA_DB_DIR = "chroma_db"
vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)

# ============ STREAMLIT UI ==============
st.title("🔍 Local Multi-Agent RAG Chatbot")

query = st.text_input("💬 Enter your query:")

if query:
    with st.spinner("🔎 Retrieving documents..."):
        retrieved_docs = vectorstore.similarity_search(query, k=3)
        doc_texts = " ".join([doc.page_content for doc in retrieved_docs]) if retrieved_docs else "No relevant documents found."

    with st.spinner("🤖 Generating response..."):
        prompt = f"You are a helpful AI assistant. Answer in a conversational way.\nContext: {doc_texts}\nUser: {query}\nAI:"
        
        # Create an empty placeholder for response
        response_container = st.empty()
        full_response = ""

        for chunk in llm(prompt, max_tokens=256, stream=True):
            full_response += chunk["choices"][0]["text"]
            response_container.write(full_response)  # Update the same text block instead of creating new lines

    # Debugging: Print response to console
    print(full_response)

