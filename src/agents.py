# agents.py
import os
from langchain.llms import HuggingFaceHub
from langchain.agents import initialize_agent, Tool

# Ensure Hugging Face API token is set via environment variables
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not HUGGINGFACEHUB_API_TOKEN:
    raise ValueError("Hugging Face API token is missing. Set 'HUGGINGFACEHUB_API_TOKEN' in Railway.")

# Load the Hugging Face model via LangChain
try:
    llm = HuggingFaceHub(repo_id="mistralai/Mistral-7B-Instruct-v0.3", model_kwargs={"temperature": 0.7})
except Exception as e:
    print(f"Error loading Hugging Face model: {e}")
    raise RuntimeError("Failed to load Hugging Face model.")

# Define a custom tool (for example, a simple processing tool)
def custom_tool(query: str) -> str:
    # Here, you could integrate a retrieval function or any custom logic
    return f"Processed query: {query}"

tool = Tool(
    name="CustomTool",
    func=custom_tool,
    description="A custom tool to process queries."
)

# Initialize an agent with the tool; using a simple zero-shot agent for demonstration
try:
    agent = initialize_agent(
        tools=[tool],
        llm=llm,
        agent="zero-shot-react-description",
        verbose=True,
        handle_parsing_errors=True  # Added this parameter
    )
except Exception as e:
    print(f"Error initializing agent: {e}")
    raise RuntimeError("Failed to initialize the agent.")

def get_agent_response(query: str) -> str:
    """Use the agent to generate a response based on the query."""
    try:
        return agent.run(query)
    except Exception as e:
        print(f"Error running agent: {e}")
        return "Error processing request."
