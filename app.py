from fastapi import FastAPI
from langchain.llms import HuggingFaceHub
import os

app = FastAPI()

os.environ["HF_API_TOKEN"] = "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"


llm = HuggingFaceHub(repo_id="google/flan-t5-small")

@app.get("/")
async def read_root():
    return {"message": "Hello, Multi-Agent RAG with Hugging Face!"}
@app.get("/ask")
async def ask(query: str):
    print(f"/ask endpoint called with query: {query}")
    return {"response": f"You asked: {query}"}

