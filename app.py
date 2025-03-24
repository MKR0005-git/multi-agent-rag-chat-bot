import os
import streamlit as st
from llama_cpp import Llama
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

# ============ SETUP LOCAL LLM ============

MODEL_PATH = r"C:\Users\kedha\multi-agent-rag-chat-bot\models\mistral-7b-instruct-v0.2.Q4_K_M.gguf"

# Load LLaMA/Mistral model with optimized parameters
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,  # Reduce context length for better speed
    n_batch=128,  # Reduce batch size to optimize memory usage
    n_threads=os.cpu_count(),  # Use all CPU cores for better performance
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
        
        response = ""
        for chunk in llm(prompt, max_tokens=256, stream=True):  # Stream output
            response += chunk["choices"][0]["text"]
            st.write(response)  # Display partial response in real time

    # Debugging: Print response to console
    print(response)
