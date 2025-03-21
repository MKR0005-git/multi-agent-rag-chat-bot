from fastapi import FastAPI, Query
from langchain.llms import HuggingFaceHub
import os

app = FastAPI()

# Set Hugging Face API Key
os.environ["HF_API_TOKEN"] = "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"

# Load model from Hugging Face
llm = HuggingFaceHub(repo_id="google/flan-t5-small")

@app.get("/")
async def read_root():
    return {"message": "Hello, Multi-Agent RAG with Hugging Face!"}


@app.get("/ask")
async def ask_question(query: str = Query(..., description="Your query")):
    response = llm(query)
    return {"response": response}
