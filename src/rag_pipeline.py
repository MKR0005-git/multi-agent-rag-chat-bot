import chromadb
from chromadb.config import Settings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
import os
from agents import get_agent_response  # Import the agent response function

# Initialize the embedding model (using a sentence transformer)
try:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
except Exception as e:
    print(f"Error loading embedding model: {e}")
    raise RuntimeError("Failed to load embedding model.")

# Define the directory where ChromaDB will store its data
persist_directory = os.getenv("CHROMA_DB_DIR", "chroma_db")  # Set via environment variable

# ChromaDB settings
try:
    client_settings = Settings(
        persist_directory=persist_directory,
        anonymized_telemetry=False  # Disable telemetry if desired
    )
    chroma_client = chromadb.PersistentClient(path=persist_directory)
except Exception as e:
    print(f"Error initializing ChromaDB: {e}")
    raise RuntimeError("Failed to initialize ChromaDB.")

# Sample documents to add to the vector store (Replace with actual data)
documents = [
    "Document 1 text content.",
    "Document 2 text content.",
    "Document 3 text content."
]
metadatas = [{"source": "doc1"}, {"source": "doc2"}, {"source": "doc3"}]
ids = ["doc1", "doc2", "doc3"]

# Create or load a vectorstore using Chroma
try:
    vectorstore = Chroma.from_texts(
        texts=documents, 
        embedding=embeddings, 
        metadatas=metadatas, 
        ids=ids, 
        client=chroma_client
    )
except Exception as e:
    print(f"Error initializing vectorstore: {e}")
    raise RuntimeError("Failed to initialize vectorstore.")

def retrieve_documents(query: str, k: int = 2):
    """
    Perform a similarity search on the vectorstore using the given query.
    Returns the top 'k' documents.
    """
    try:
        docs = vectorstore.similarity_search(query, k=k)
        return docs
    except Exception as e:
        print(f"Error retrieving documents: {e}")
        return []

def query_with_agent(query: str):
    """
    Retrieve documents and pass them to the agent for further processing and response generation.
    """
    retrieved_docs = retrieve_documents(query, k=3)  # Retrieve top 3 documents
    # You can further process the documents or format them before passing to the agent
    doc_texts = [doc.page_content for doc in retrieved_docs]

    # Format query and documents to pass to the agent
    formatted_query = f"Here are some documents: {doc_texts}. Query: {query}"

    # Get the agent's response using the formatted query
    response = get_agent_response(formatted_query)
    return response
