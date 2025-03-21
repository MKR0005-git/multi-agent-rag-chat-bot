from fastapi import FastAPI
from langchain.llms import HuggingFaceHub
import os

app = FastAPI()

os.environ["HF_API_TOKEN"] = "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"
print(os.getenv("HF_API_TOKEN"))

llm = HuggingFaceHub(repo_id="google/flan-t5-small")

@app.get("/")
async def read_root():
    return {"message": "Hello, Multi-Agent RAG with Hugging Face!"}

@app.get("/ask")
async def ask(query: str):
    return {"response": f"You asked: {query}"}
