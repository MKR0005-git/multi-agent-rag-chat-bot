from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub  # Correct import path
from langchain.prompts import PromptTemplate
from sentence_transformers import CrossEncoder

# Initialize FastAPI
app = FastAPI()

# Load Embeddings Model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Simulated FAISS database (Replace with actual FAISS index)
vector_db = FAISS.from_texts(["Example document 1", "Example document 2"], embeddings)

# Define retriever (Limit to top 3 documents)
retriever = vector_db.as_retriever(search_kwargs={"k": 3})

# Initialize Cross-Encoder for Re-Ranking
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank_documents(query, retrieved_docs):
    """Re-rank retrieved documents using a cross-encoder model."""
    pairs = [(query, doc) for doc in retrieved_docs]  # Pair query with each doc
    scores = reranker.predict(pairs)  # Get relevance scores
    ranked_docs = [doc for _, doc in sorted(zip(scores, retrieved_docs), reverse=True)]
    return ranked_docs

# Define LLM (No API Key Here)
llm = HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    model_kwargs={"temperature": 0.7, "max_length": 256}
)

# Define Prompt Template
prompt_template = PromptTemplate(
    input_variables=["context", "query"],
    template="Answer the query based on the context provided: {query}\n\nContext: {context}"
)

# Query Handling Function
def query_rag_system(query):
    # Retrieve top 3 documents from FAISS
    retrieved_docs = vector_db.similarity_search(query, k=3)


    # Extract text content and re-rank
    ranked_docs = rerank_documents(query, [doc.page_content for doc in retrieved_docs])

    # Construct context from top-ranked documents
    context = "\n\n".join(ranked_docs)

    # Format prompt
    prompt = prompt_template.format(context=context, query=query)

    # Generate response from LLM
    response = llm(prompt)

    return response

# API Endpoint
class QueryRequest(BaseModel):
    query: str

@app.post("/query")
def query_endpoint(request: QueryRequest):
    response = query_rag_system(request.query)
    return {"query": request.query, "response": response}
