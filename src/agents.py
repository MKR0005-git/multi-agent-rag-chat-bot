import os
from langchain.llms import HuggingFaceHub
from langchain.agents import initialize_agent, Tool

# Set your Hugging Face API token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# Load the Hugging Face model via LangChain
llm = HuggingFaceHub(repo_id="mistralai/Mistral-7B-Instruct-v0.3", model_kwargs={"temperature": 0.7})

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
agent = initialize_agent(
    tools=[tool],
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True,
    handle_parsing_errors=True  # Added this parameter
)


def get_agent_response(query: str) -> str:
    # Use the agent to generate a response based on the query
    return agent.run(query)
