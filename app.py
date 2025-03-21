from fastapi import FastAPI
from langchain.llms import HuggingFaceHub
import os

app = FastAPI()

# Set your Hugging Face API key
os.environ["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"] = "your_huggingface_api_key"

# Load a model from Hugging Face
llm = HuggingFaceHub(repo_id="google/flan-t5-small")

@app.get("/")
def read_root():
    return {"message": "Hello, Multi-Agent RAG with Hugging Face!"}
@app.get("/ask")
def ask_question(query: str = ""):  # Default empty query to avoid errors
    if not query:
        return {"error": "Query parameter is required"}
    response = llm(query)
    return {"response": response}
