from fastapi import FastAPI
from pydantic import BaseModel
import requests
import os
import logging

# Initialize FastAPI app
app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define request model
class QueryRequest(BaseModel):
    query: str

# Get Hugging Face API Key from environment variables
HF_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
HF_MODEL_API = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"

# Ensure API Key is available
if not HF_API_TOKEN:
    logger.error("Missing Hugging Face API Key. Set HUGGINGFACEHUB_API_TOKEN in environment variables.")

@app.post("/query")
def process_query(request: QueryRequest):
    """
    Process user query and fetch response from Hugging Face model.
    """
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}", "Content-Type": "application/json"}
    payload = {"inputs": request.query}

    try:
        response = requests.post(HF_MODEL_API, json=payload, headers=headers)
        response.raise_for_status()  # Raise error if API call fails
        
        response_data = response.json()

        # Handle different response formats
        if isinstance(response_data, list) and len(response_data) > 0:
            ai_response = response_data[0].get("generated_text", "No response generated.")
        elif isinstance(response_data, dict) and "error" in response_data:
            ai_response = f"Error from API: {response_data['error']}"
        else:
            ai_response = "Unexpected response format."

        logger.info(f"Query: {request.query} | AI Response: {ai_response}")
        return {"query": request.query, "response": ai_response}

    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching AI response: {e}")
        return {"query": request.query, "response": "Sorry, there was an error processing your request."}
