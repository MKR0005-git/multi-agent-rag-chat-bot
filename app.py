import os
from langchain.agents import AgentType, initialize_agent
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
from fastapi import FastAPI, Query
from huggingface_hub import InferenceClient
from src.rag_pipeline import retrieve_documents  # Ensure your retrieval function is correctly imported

# Load Hugging Face API Key (Use Correct Name)
HF_API_KEY = os.getenv("HUGGINGFACEHUB_API_TOKEN")  # Ensure it's set in your environment

# Initialize Hugging Face Client
MODEL_NAME = "google/flan-t5-small"
hf_client = InferenceClient(MODEL_NAME, token=HF_API_KEY)

# Retrieval Function
def retrieve_and_format(query: str):
    documents = retrieve_documents(query)
    context = "\n".join([doc["content"] for doc in documents])
    return f"Context:\n{context}\n\nQuery: {query}"

# Hugging Face Inference Function
def query_llm(prompt: str):
    response = hf_client.text_generation(prompt, max_new_tokens=200)
    return response.strip()

# Define Tool for Retrieval
retrieval_tool = Tool(
    name="RetrieveDocuments",
    func=retrieve_and_format,
    description="Retrieves relevant documents given a query."
)

# Agent Memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Initialize Agent (Using Only Hugging Face)
agent_executor = initialize_agent(
    tools=[retrieval_tool],
    llm=LLMChain(
        llm=lambda prompt: query_llm(prompt),  # Uses Hugging Face API
        prompt=PromptTemplate.from_template("{chat_history}\n{query}"),
        memory=memory
    ),
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Create FastAPI App
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Multi-Agent RAG Chatbot is running!"}

@app.get("/retrieve")
def retrieve(query: str = Query(..., description="Query for document retrieval")):
    documents = retrieve_documents(query)
    return {"documents": documents}

@app.get("/ask")
def ask(query: str = Query(..., description="User query for chatbot")):
    try:
        retrieved_context = retrieve_and_format(query)
        response = query_llm(retrieved_context)
        return {"response": response}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
