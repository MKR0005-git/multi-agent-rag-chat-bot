import chromadb
from chromadb.config import Settings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

# Initialize the embedding model (you can choose another model if desired)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize Chroma vector store
persist_directory = "chroma_db"
chroma_client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=persist_directory))

# Assume you have some documents to add:
documents = [
    "Document 1 text content.",
    "Document 2 text content.",
    "Document 3 text content."
]
metadatas = [{"source": "doc1"}, {"source": "doc2"}, {"source": "doc3"}]
ids = ["doc1", "doc2", "doc3"]

vectorstore = Chroma.from_texts(texts=documents, embedding=embeddings, metadatas=metadatas, ids=ids, client=chroma_client)

# Define a simple retrieval function
def retrieve_documents(query: str, k: int = 2):
    docs = vectorstore.similarity_search(query, k=k)
    return docs
