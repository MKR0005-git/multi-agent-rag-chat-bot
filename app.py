import os
from fastapi import FastAPI

app = FastAPI()

# Set your Hugging Face API token here (replace with your actual token)
os.environ["HF_API_TOKEN"] = "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"

@app.get("/")
async def read_root():
    return {"message": "Hello, Multi-Agent RAG with Hugging Face!"}

@app.get("/ask")
async def ask(query: str):
    return {"response": f"You asked: {query}"}
