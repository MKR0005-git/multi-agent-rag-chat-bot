from fastapi import FastAPI
from src.agents import get_agent_response  # Import your multi-agent response function

app = FastAPI()

@app.get("/")
async def read_root():
    return {"message": "Hello, Multi-Agent RAG with Hugging Face!"}

@app.get("/ask")
async def ask(query: str):
    response = get_agent_response(query)
    return {"response": response}
