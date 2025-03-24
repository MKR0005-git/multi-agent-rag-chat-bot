import os
from llama_cpp import Llama

# Setup Local LLM
MODEL_PATH = r"C:\Users\kedha\multi-agent-rag-chat-bot\models\mistral-7b-instruct-v0.2.Q4_K_M.gguf"  # Ensure correct model path

# Load LLaMA/Mistral model
try:
    llm = Llama(model_path=MODEL_PATH, n_ctx=2048, n_batch=256)
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit(1)

def generate_response(agent_prompt, user_query):
    """Generate a response using the local LLM."""
    prompt = f"{agent_prompt}\nUser Query: {user_query}"
    response = llm(prompt)
    
    if "choices" in response and response["choices"]:
        return response["choices"][0]["text"].strip()
    return "⚠️ No response generated."

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
    agents = {
        "research": ResearchAgent(),
        "summary": SummaryAgent(),
        "creative": CreativeAgent(),
    }
    return agents.get(agent_type, None)

# Testing the agent system
if __name__ == "__main__":
    agent_type = "research"  # Change to "summary" or "creative" for different agents
    query = "Explain quantum computing in simple terms."
    
    agent = get_agent(agent_type)
    if agent:
        print(f"🔹 {agent_type.capitalize()} Agent Response:\n", agent.respond(query))
    else:
        print("⚠️ Invalid agent type selected.")
