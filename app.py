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

# Hugging Face API Configuration
HF_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
HF_MODEL_API = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"

# Headers for API request
HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}

@app.post("/query")
def process_query(request: QueryRequest):
    """
    Process user query and fetch response from Hugging Face model.
    """
    payload = {"inputs": request.query}

    try:
        response = requests.post(HF_MODEL_API, json=payload, headers=HEADERS)
        response.raise_for_status()  # Raise error if API fails

        # Extract response
        data = response.json()
        if isinstance(data, list) and "generated_text" in data[0]:
            ai_response = data[0]["generated_text"]
        else:
            ai_response = "Sorry, I couldn't generate a response."

        logger.info(f"Query: {request.query} | AI Response: {ai_response}")
        return {"query": request.query, "response": ai_response}

    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching AI response: {e}")
        return {"query": request.query, "response": "Sorry, there was an error processing your request."}
