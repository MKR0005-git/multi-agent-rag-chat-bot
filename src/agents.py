from llama_cpp import Llama

# Load Local LLaMA model
MODEL_PATH = "path/to/your/model.gguf"
llm = Llama(model_path=MODEL_PATH, n_ctx=2048, n_batch=256)

class Agent:
    """Base class for all agents."""
    def __init__(self, name):
        self.name = name

    def run(self, query, context=""):
        """Process the query with context using LLM."""
        prompt = f"{context}\n{query}"
        response = llm(prompt)
        return response["choices"][0]["text"]

class ResearchAgent(Agent):
    """Agent specialized in research and information retrieval."""
    def run(self, query, context=""):
        return super().run(f"Provide a detailed research answer: {query}", context)

class SummarizerAgent(Agent):
    """Agent specialized in summarization."""
    def run(self, query, context=""):
        return super().run(f"Summarize this information: {context}", query)

# Example usage
if __name__ == "__main__":
    research_agent = ResearchAgent("Researcher")
    summarizer_agent = SummarizerAgent("Summarizer")

    query = "Explain quantum computing in simple terms."
    research_response = research_agent.run(query)
    summary_response = summarizer_agent.run(query, research_response)

    print("Research Response:", research_response)
    print("Summary Response:", summary_response)
