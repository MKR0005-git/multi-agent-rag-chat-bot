import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document
from llama_cpp import Llama

# ============ Load Local LLaMA Model ============
MODEL_PATH = "path/to/your/model.gguf"
llm = Llama(model_path=MODEL_PATH, n_ctx=2048, n_batch=256)

# ============ Setup ChromaDB (Vector Store) ============
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
CHROMA_DB_DIR = "chroma_db"
vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)

class RAGPipeline:
    """Retrieval-Augmented Generation pipeline using local LLM and ChromaDB."""
    
    def retrieve_documents(self, query, k=3):
        """Retrieve relevant documents from ChromaDB."""
        return vectorstore.similarity_search(query, k=k)

    def generate_answer(self, query):
        """Retrieve documents and generate an answer using the LLM."""
        docs = self.retrieve_documents(query)
        context = " ".join([doc.page_content for doc in docs])

        prompt = f"Here is relevant information: {context}\n\nNow answer this query: {query}"
        response = llm(prompt)

        return response["choices"][0]["text"]

# Example usage
if __name__ == "__main__":
    rag = RAGPipeline()
    query = "What are the applications of artificial intelligence in healthcare?"
    
    response = rag.generate_answer(query)
    print("Generated Answer:", response)
