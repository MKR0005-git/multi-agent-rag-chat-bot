import chromadb
from chromadb.config import Settings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

# Initialize the embedding model (using a sentence transformer)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Define the directory where Chroma will store its data
persist_directory = "chroma_db"

# Configure Chroma settings using the new API
client_settings = Settings(chroma_db_impl="duckdb+parquet", persist_directory=persist_directory)
chroma_client = chromadb.Client(settings=client_settings)

# Sample documents to add to the vector store
documents = [
    "Document 1 text content.",
    "Document 2 text content.",
    "Document 3 text content."
]
metadatas = [{"source": "doc1"}, {"source": "doc2"}, {"source": "doc3"}]
ids = ["doc1", "doc2", "doc3"]

# Create a vectorstore using Chroma
vectorstore = Chroma.from_texts(
    texts=documents, 
    embedding=embeddings, 
    metadatas=metadatas, 
    ids=ids, 
    client=chroma_client
)

def retrieve_documents(query: str, k: int = 2):
    """
    Perform a similarity search on the vectorstore using the given query.
    Returns the top 'k' documents.
    """
    docs = vectorstore.similarity_search(query, k=k)
    return docs
