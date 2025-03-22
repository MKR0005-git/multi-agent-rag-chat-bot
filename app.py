from fastapi import FastAPI
from src.agents import get_agent_response  # Multi-agent response function
from src.rag_pipeline import retrieve_documents  # Document retrieval function

app = FastAPI()

@app.get("/")
async def read_root():
    return {"message": "Multi-Agent RAG Chatbot is running!"}

@app.get("/ask")
async def ask(query: str):
    # Retrieve relevant documents
    docs = retrieve_documents(query, k=2)
    context = "\n".join([doc.page_content for doc in docs])
    combined_query = f"{query}\nContext:\n{context}"
    response = get_agent_response(combined_query)
    return {"response": response}
