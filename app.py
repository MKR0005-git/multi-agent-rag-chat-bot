from fastapi import FastAPI
from pydantic import BaseModel
import requests
import logging
import os

# Initialize FastAPI app
app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define request model
class QueryRequest(BaseModel):
    query: str

# Load Hugging Face API key from environment variable
API_KEY = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not API_KEY:
    logger.error("❌ Missing Hugging Face API Key. Set HUGGINGFACEHUB_API_TOKEN in environment variables.")
    raise ValueError("Missing API Key. Set HUGGINGFACEHUB_API_TOKEN.")

# Hugging Face Model API Endpoint (Update with your actual model)
AI_MODEL_API = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"


@app.post("/query")
def process_query(request: QueryRequest):
    """
    Process user query and fetch response from Hugging Face model.
    """
    payload = {
        "inputs": request.query,
        "parameters": {"temperature": 0.7, "max_new_tokens": 100}
    }

    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

    try:
        response = requests.post(AI_MODEL_API, json=payload, headers=headers)
        response.raise_for_status()  # Ensure request was successful

        ai_response = response.json()
        generated_text = ai_response[0].get("generated_text", "❌ Error: No response generated.")

        logger.info(f"✅ Query: {request.query} | AI Response: {generated_text}")
        return {"query": request.query, "response": generated_text}

    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Error fetching AI response: {e}")
        return {"query": request.query, "response": "❌ Sorry, there was an error processing your request."}
