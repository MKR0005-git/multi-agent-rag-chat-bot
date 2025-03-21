from fastapi import FastAPI
from langchain.llms import HuggingFaceHub
import os

app = FastAPI()

# Set Hugging Face API Key
os.environ["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"] = "your_huggingface_api_key"

# Load model from Hugging Face
llm = HuggingFaceHub(repo_id="google/flan-t5-small")

@app.get("/")
def read_root():
    return {"message": "Hello, Multi-Agent RAG with Hugging Face!"}

@app.get("/ask")
def ask_question(query: str):
    response = llm(query)
    return {"response": response}

