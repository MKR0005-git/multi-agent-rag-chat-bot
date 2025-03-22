import os
import uvicorn
import pickle
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from sentence_transformers import CrossEncoder

app = FastAPI()

# Enable CORS (Required for external access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update to specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Embeddings Model
try:
    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
except Exception as e:
    print(f"Error loading embeddings: {e}")
    raise RuntimeError("Failed to load embeddings model.")

# Persistent FAISS Database Setup
FAISS_INDEX_PATH = "faiss_index.pkl"

try:
    if os.path.exists(FAISS_INDEX_PATH):
        print("Loading existing FAISS index...")
        with open(FAISS_INDEX_PATH, "rb") as f:
            vector_db = pickle.load(f)
    else:
        print("Initializing FAISS database...")
        vector_db = FAISS.from_texts(["Example document 1", "Example document 2"], embeddings)
        with open(FAISS_INDEX_PATH, "wb") as f:
            pickle.dump(vector_db, f)
except Exception as e:
    print(f"Error initializing FAISS: {e}")
    raise RuntimeError("Failed to initialize FAISS database.")

# Define retriever (Limit to top 3 documents)
retriever = vector_db.as_retriever(search_kwargs={"k": 3})

# Initialize Cross-Encoder for Re-Ranking
try:
    print("Loading Cross-Encoder model...")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
except Exception as e:
    print(f"Error loading Cross-Encoder model: {e}")
    raise RuntimeError("Failed to load Cross-Encoder model.")

def rerank_documents(query, retrieved_docs):
    """Re-rank retrieved documents using a cross-encoder model."""
    if not retrieved_docs:
        return []
    
    pairs = [(query, doc) for doc in retrieved_docs]
    scores = reranker.predict(pairs)
    ranked_docs = [doc for _, doc in sorted(zip(scores, retrieved_docs), reverse=True)]
    return ranked_docs

# Define LLM Model with HuggingFaceEndpoint
try:
    print("Loading LLM model...")
    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.3",
        temperature=0.3,
        model_kwargs={"max_length": 50}
    )
except Exception as e:
    print(f"Error loading LLM model: {e}")
    raise RuntimeError("Failed to load LLM model.")

# Fixed Prompt Template
prompt_template = PromptTemplate(
    input_variables=["query"],
    template="Provide a brief and direct answer to the question.\n\nQuestion: {query}\n\nAnswer:"
)

def query_rag_system(query):
    """Retrieve, re-rank, and generate response from LLM."""
    try:
        retrieved_docs = vector_db.similarity_search(query, k=3)
        ranked_docs = rerank_documents(query, [doc.page_content for doc in retrieved_docs])
        prompt = prompt_template.format(query=query)
        raw_response = llm.invoke(prompt)
        response = raw_response.strip().split("\n")[0]
        return response
    except Exception as e:
        print(f"Error processing query: {e}")
        return "Error processing request."

# API Endpoint
class QueryRequest(BaseModel):
    query: str

@app.post("/query")
def query_endpoint(request: QueryRequest):
    response = query_rag_system(request.query)
    if response == "Error processing request.":
        raise HTTPException(status_code=500, detail="Internal Server Error")
    return {"query": request.query, "response": response}

# Start FastAPI on Railway
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))  # Railway assigns PORT dynamically
    print(f"Starting server on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)
