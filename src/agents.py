import os
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from llama_cpp import Llama

# Setup Local LLM
MODEL_PATH = "models/mistral-7b-instruct.gguf"  # Update path
llm = Llama(model_path=MODEL_PATH, n_ctx=2048, n_batch=256)

def generate_response(agent_prompt, user_query):
    """Generate a response using the local LLM."""
    prompt = f"{agent_prompt}\nUser Query: {user_query}"
    response = llm(prompt)
    return response["choices"][0]["text"].strip()

# Define specialized agents
class ResearchAgent:
    def respond(self, query):
        prompt = "You are a research assistant. Provide detailed, well-cited answers."
        return generate_response(prompt, query)

class SummaryAgent:
    def respond(self, query):
        prompt = "Summarize key points in a concise and clear way."
        return generate_response(prompt, query)

class CreativeAgent:
    def respond(self, query):
        prompt = "Be creative and generate engaging, thought-provoking responses."
        return generate_response(prompt, query)

# Agent selection
def get_agent(agent_type):
    if agent_type == "research":
        return ResearchAgent()
    elif agent_type == "summary":
        return SummaryAgent()
    elif agent_type == "creative":
        return CreativeAgent()
    else:
        raise ValueError("Unknown agent type")
