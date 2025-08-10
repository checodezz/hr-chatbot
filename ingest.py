import os
import json
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Validate required environment variables
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY environment variable is required")

# Path to JSON file
DATA_FILE = Path("data/employees.json")

# Load employee data from JSON
with open(DATA_FILE, "r", encoding="utf-8") as f:
    employees = json.load(f)["employees"]  # if JSON has a top-level "employees" key

# Qdrant connection/config from environment variables
COLLECTION = os.getenv("QDRANT_COLLECTION", "employees_rag")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))

# Initialize Qdrant client and embeddings
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


def employee_to_document(emp):
    """Converts a single employee profile into a LangChain Document."""
    content = (
        f"Employee: {emp['name']}. "
        f"Skills: {', '.join(emp['skills'])}. "
        f"Experience: {emp['experience_years']} years. "
        f"Projects: {', '.join(emp['projects'])}. "
        f"Availability: {emp['availability']}."
    )
    metadata = {
        "id": emp["id"],
        "name": emp["name"],
        "skills": emp["skills"],
        "experience_years": emp["experience_years"],
        "projects": emp["projects"],
        "availability": emp["availability"]
    }
    return Document(page_content=content, metadata=metadata)

def ingest_employees():
    # Convert employees to documents
    docs = [employee_to_document(emp) for emp in employees]

    # Delete existing collection (avoids duplicates)
    try:
        client.delete_collection(collection_name=COLLECTION)
        print(f"Deleted existing collection: {COLLECTION}")
    except Exception as e:
        print(f"No previous collection or could not delete: {e}")

    # Get vector dimension by generating a sample embedding
    sample_vector = embeddings.embed_query("sample text")
    vector_size = len(sample_vector)
    print(f"Vector dimension: {vector_size}")

    # Create collection with proper vector configuration
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
    )
    print(f"Created collection: {COLLECTION}")

    # Generate embeddings & prepare points
    points = []
    for i, doc in enumerate(docs):
        print(f"Processing employee {i+1}/{len(docs)}: {doc.metadata['name']}")
        vector = embeddings.embed_query(doc.page_content)
        points.append({
            "id": doc.metadata["id"],
            "vector": vector,
            "payload": {
                "content": doc.page_content,
                **doc.metadata
            }
        })

    # Upload to Qdrant
    client.upsert(collection_name=COLLECTION, points=points)
    print(f"Successfully ingested {len(points)} employees into collection '{COLLECTION}'")

if __name__ == "__main__":
    ingest_employees()