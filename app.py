from fastapi import FastAPI
from src.agents import get_agent_response  # your agent logic
from src.rag_pipeline import retrieve_documents  # import the retrieval function

app = FastAPI()

@app.get("/")
async def read_root():
    return {"message": "Hello, Multi-Agent RAG with Hugging Face!"}

@app.get("/ask")
async def ask(query: str):
    # You can later integrate retrieval into your agent logic here
    response = get_agent_response(query)
    return {"response": response}

@app.get("/retrieve")
async def retrieve(query: str):
    docs = retrieve_documents(query)
    # Optionally, format the output for clarity
    formatted_docs = [{"id": doc.metadata.get("source", "unknown"), "content": doc.page_content} for doc in docs]
    return {"documents": formatted_docs}
