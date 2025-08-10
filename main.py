import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from rag_chain import get_rag_chain, run_query
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Employee RAG API", description="Query employee data using semantic search")

# Configuration from environment variables
COLLECTION = os.getenv("QDRANT_COLLECTION", "employees_rag")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))

# Validate required environment variables
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY environment variable is required")

# Initialize clients
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION,
    embedding=embeddings,
    content_payload_key="content",  # Specify the payload key for content
    metadata_payload_key=None,  # Don't use separate metadata key
)

# Initialize the RAG chain
rag_chain = get_rag_chain(vector_store)

# Pydantic models for request/response
class QueryRequest(BaseModel):
    query: str
    k: Optional[int] = 5  # Number of results to return
    filter_availability: Optional[str] = None  # "available", "on leave", "on project"
    min_experience: Optional[int] = None
    system_prompt: Optional[str] = None  # Custom system prompt

class QueryResponse(BaseModel):
    query: str
    llm_response: str
    sources: List[str]  # keep track of where info came from

class CustomPromptRequest(BaseModel):
    query: str
    system_prompt: str

@app.get("/")
async def root():
    return {"message": "Employee RAG API is running"}

@app.get("/health")
async def health_check():
    try:
        # Check if Qdrant collection exists
        collection_info = client.get_collection(COLLECTION)
        return {
            "status": "healthy",
            "collection": COLLECTION,
            "vector_count": collection_info.vectors_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def query_employees(request: QueryRequest):
    try:
        # Create RAG chain with custom system prompt if provided
        if request.system_prompt:
            custom_rag_chain = get_rag_chain(vector_store, system_prompt=request.system_prompt)
            llm_answer, source_docs = run_query(request.query, custom_rag_chain)
        else:
            # Use default RAG chain
            llm_answer, source_docs = run_query(request.query, rag_chain)

        # Convert source documents to strings for the API output
        sources_list = [doc.page_content for doc in source_docs]

        return QueryResponse(
            query=request.query,
            llm_response=llm_answer,
            sources=sources_list
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@app.post("/query/simple")
async def simple_query(query: str, k: int = 5):
    """
    Simple query endpoint that accepts query as a query parameter.
    """
    request = QueryRequest(query=query, k=k)
    return await query_employees(request)

@app.get("/employees/available")
async def get_available_employees():
    """Get all available employees."""
    request = QueryRequest(
        query="available employee",
        filter_availability="available",
        k=50
    )
    return await query_employees(request)

@app.get("/employees/skills/{skill}")
async def get_employees_by_skill(skill: str, available_only: bool = False):
    """Find employees with a specific skill."""
    request = QueryRequest(
        query=f"employee with {skill} skills",
        filter_availability="available" if available_only else None,
        k=20
    )
    return await query_employees(request)

@app.post("/query/custom-prompt")
async def query_with_custom_prompt(request: CustomPromptRequest):
    """
    Query endpoint that accepts a custom system prompt.
    Useful for testing different response formats and behaviors.
    """
    query_request = QueryRequest(
        query=request.query,
        system_prompt=request.system_prompt
    )
    return await query_employees(query_request)

if __name__ == "__main__":
    import uvicorn
    app_host = os.getenv("APP_HOST", "0.0.0.0")
    app_port = int(os.getenv("APP_PORT", "8000"))
    uvicorn.run(app, host=app_host, port=app_port)