from fastapi import FastAPI, Query
from langchain_community.llms import HuggingFaceHub
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from pydantic import BaseModel
import os

# Initialize FastAPI app
app = FastAPI()

# Hugging Face API Key (Ensure it's set in your environment)
HF_API_KEY = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Load Hugging Face LLM
llm = HuggingFaceHub(
    repo_id="google/flan-t5-small",
    model_kwargs={"temperature": 0.7, "max_length": 256},
    huggingfacehub_api_token=HF_API_KEY
)

# Load embeddings model for retrieval
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Simulated FAISS database (replace with actual FAISS index)
vector_db = FAISS.from_texts(["Example document 1", "Example document 2"], embeddings)

# Define Prompt Template for LLM
prompt_template = PromptTemplate(
    input_variables=["context", "query"],
    template="Given the context: {context}\nAnswer the query: {query}"
)

# LLM Chain
llm_chain = LLMChain(prompt=prompt_template, llm=llm)

class QueryModel(BaseModel):
    query: str

@app.get("/")
def read_root():
    return {"message": "Multi-Agent RAG Chatbot is Running!"}

@app.get("/retrieve")
def retrieve(query: str = Query(..., title="Query")):
    """Retrieves relevant documents from FAISS DB."""
    docs = vector_db.similarity_search(query, k=2)
    return {"documents": [{"id": f"doc{i+1}", "content": doc.page_content} for i, doc in enumerate(docs)]}

@app.get("/ask")
def ask(query: str = Query(..., title="User Query")):
    """Generates response using retrieved context and LLM."""
    retrieved_docs = vector_db.similarity_search(query, k=2)
    context = " ".join([doc.page_content for doc in retrieved_docs])
    
    # Generate Response
    response = llm_chain.run({"context": context, "query": query})
    
    return {"query": query, "response": response}

