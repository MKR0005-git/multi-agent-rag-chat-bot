import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from llama_cpp import Llama

# ========== Setup Local LLM ==========
MODEL_PATH = "models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"  # Updated to match the actual model file
llm = Llama(model_path=MODEL_PATH, n_ctx=2048, n_batch=256)

# ========== Setup ChromaDB ==========
CHROMA_DB_DIR = "chroma_db"
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)

def retrieve_documents(query, k=3):
    """Retrieve relevant documents from the vector database."""
    retrieved_docs = vectorstore.similarity_search(query, k=k)
    return " ".join([doc.page_content for doc in retrieved_docs])

def generate_rag_response(query):
    """Generate response using retrieved context and local LLM."""
    context = retrieve_documents(query)
    prompt = f"Context: {context}\nUser Query: {query}\nAnswer:"
    response = llm(prompt)
    return response["choices"][0]["text"].strip()
