from fastapi import FastAPI
from pydantic import BaseModel
import requests
import logging

# Initialize FastAPI app
app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define request model
class QueryRequest(BaseModel):
    query: str

# External AI model API (Replace with your actual AI service URL)
AI_MODEL_API = "https://api.openai.com/v1/chat/completions"  # Example OpenAI API

# API Key (Replace with your actual API key)
API_KEY = "your_openai_api_key"

# Custom system prompt to enhance responses
SYSTEM_PROMPT = (
    "You are an intelligent AI assistant with deep knowledge. "
    "Provide structured, detailed, and meaningful responses. "
    "Ensure clarity, factual accuracy, and coherence."
)

@app.post("/query")
def process_query(request: QueryRequest):
    """
    Process user query and fetch response from AI model.
    """
    user_prompt = f"User Query: {request.query}\nAssistant: "

    payload = {
        "model": "gpt-4",  # Replace with actual model name
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.7
    }

    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

    try:
        response = requests.post(AI_MODEL_API, json=payload, headers=headers)
        response.raise_for_status()  # Raise error if API fails
        ai_response = response.json()["choices"][0]["message"]["content"]
        logger.info(f"Query: {request.query} | AI Response: {ai_response}")
        return {"query": request.query, "response": ai_response}

    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching AI response: {e}")
        return {"query": request.query, "response": "Sorry, there was an error processing your request."}
