# HR Chatbot - RAG Pipeline

A complete RAG (Retrieval-Augmented Generation) pipeline for querying employee data using semantic search.

## Features

- 🔍 Semantic search of employee profiles
- 🤖 AI-powered responses using GPT-3.5-turbo
- 📊 Vector storage with Qdrant
- 🌐 RESTful API with FastAPI
- 🔒 Secure environment variable management

## Setup

### 1. Clone and Install Dependencies

```bash
git clone <repository-url>
cd hr-chatbot
python -m venv venv
or
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Environment Variables

Copy the example environment file and configure your settings:

```bash
cp .env.example .env
```

Edit `.env` with your actual values:

```env
# OpenAI API Configuration
OPENAI_API_KEY=your_actual_openai_api_key_here

# Qdrant Configuration
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=employees_rag

# Application Configuration
APP_HOST=0.0.0.0
APP_PORT=8000
```

### 3. Start Qdrant

Make sure Qdrant is running on your system:

```bash
# Using Docker
docker run -p 6333:6333 qdrant/qdrant

# Or install and run locally
# Follow instructions at: https://qdrant.tech/documentation/guides/installation/
```

### 4. Ingest Data

```bash
python ingest.py
```

### 5. Start the Server

```bash
python main.py
```

The API will be available at `http://localhost:8000`

## API Endpoints

- `GET /` - Health check
- `GET /health` - Detailed system status
- `POST /query` - Main query endpoint
- `POST /query/simple` - Simple query endpoint
- `GET /employees/available` - Get available employees
- `GET /employees/skills/{skill}` - Find employees by skill

## Example Usage

```bash
# Query for Python developers
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "Find employees with Python skills"}'

# Simple query
curl -X POST "http://localhost:8000/query/simple?query=React&k=3"

# Get available employees
curl -X GET "http://localhost:8000/employees/available"
```

## Security Notes

- ✅ Never commit `.env` files to version control
- ✅ Use strong, unique API keys
- ✅ Rotate API keys regularly
- ✅ Use environment variables for all sensitive data

## Environment Variables

| Variable            | Description         | Default       | Required |
| ------------------- | ------------------- | ------------- | -------- |
| `OPENAI_API_KEY`    | Your OpenAI API key | -             | ✅       |
| `QDRANT_HOST`       | Qdrant server host  | localhost     | ❌       |
| `QDRANT_PORT`       | Qdrant server port  | 6333          | ❌       |
| `QDRANT_COLLECTION` | Collection name     | employees_rag | ❌       |
| `APP_HOST`          | API server host     | 0.0.0.0       | ❌       |
| `APP_PORT`          | API server port     | 8000          | ❌       |

## Architecture

```
Data (JSON) → Ingest → Vector Store (Qdrant) → RAG Chain → API (FastAPI)
```

1. **Data Ingestion**: Employee data converted to documents with embeddings
2. **Vector Storage**: Qdrant stores semantic vectors for fast retrieval
3. **RAG Chain**: LangChain orchestrates retrieval and generation
4. **API Layer**: FastAPI provides RESTful interface
