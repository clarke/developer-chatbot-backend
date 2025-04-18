# Developer Chatbot Backend

A powerful question-answering system that enables natural language interaction with your codebase. This application uses LangChain, Qdrant, and local language models to provide intelligent responses to questions about your code.

## Overview

This application provides a developer-friendly interface to interact with codebases using natural language. It consists of:

1. A FastAPI-based REST API that handles codebase queries
2. A vector database (Qdrant) for efficient code search and retrieval
3. Utility scripts for code ingestion and inspection

The system uses local embeddings and language models to understand and respond to questions about your codebase, making it easier to navigate and understand complex codebases. All processing is done locally, ensuring privacy and offline operation.

## Features

- Natural language querying of codebases
- Support for multiple programming languages
- Source code chunking and semantic search
- REST API for easy integration
- Docker support for containerized deployment
- Fully offline operation with local models
- High-quality embeddings and language models

## Prerequisites

- Python 3.8+
- Docker and Docker Compose
- Git (for repository ingestion)
- 8GB+ RAM (for running local models)
- 5GB+ disk space (for model storage)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd developer-chatbot-backend
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file with the following variables:
```
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=codebase
MODEL_PATH=models/mistral-7b-instruct-v0.1.Q4_K_M.gguf
```

5. Download the required models:
```bash
mkdir -p models
cd models
curl -L -o mistral-7b-instruct-v0.1.Q4_K_M.gguf https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf
```

## Usage

### Starting the Services

1. Start Qdrant using Docker Compose:
```bash
docker-compose up -d
```

2. Start the FastAPI application:
```bash
cd app
uvicorn main:app --reload
```

### Ingesting Code

Use the provided scripts to ingest code into the system:

1. Ingest a local repository:
```bash
python scripts/ingest_local_repo.py /path/to/repository [--refresh]
```
The `--refresh` flag will clear the existing collection before ingesting new data.

2. Query the codebase:
```bash
python scripts/query.py "your question here"
```

3. Inspect the Qdrant database:
```bash
python scripts/inspect_qdrant.py
```

### API Usage

The API provides the following endpoints:

- `POST /ask`: Submit questions about your codebase
- `GET /health`: Check the health status of the service

Example API request:
```bash
curl -X POST "http://localhost:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{"question": "How does the authentication system work?"}'
```

## Project Structure

```
.
├── app/                      # FastAPI application
│   ├── main.py               # Main application code
│   └── Dockerfile            # Docker configuration
├── scripts/                  # Utility scripts
│   ├── ingest_local_repo.py  # Code ingestion script
│   ├── inspect_qdrant.py     # Database inspection
│   └── query.py              # Query interface
├── models/                   # Local model storage
├── qdrant_data/              # Qdrant database storage
├── docker-compose.yml        # Docker Compose configuration
└── requirements.txt          # Python dependencies
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Testing

The application includes a comprehensive test suite that runs completely offline by mocking external dependencies (OpenAI and Qdrant).

### Running Tests

1. Install test dependencies:
```bash
pip install -r requirements-test.txt
```

2. Run the test suite:
```bash
pytest tests/
```

### Test Coverage

The test suite includes:

- `test_health_check`: Verifies the health check endpoint returns the expected response
- `test_ask_endpoint_success`: Tests successful question-answering with mocked responses
- `test_ask_endpoint_invalid_request`: Validates proper handling of invalid requests
- `test_ask_endpoint_error`: Ensures proper error handling and response formatting

### Test Dependencies

- `pytest`: Test framework
- `httpx`: Required by FastAPI TestClient
- `pytest-asyncio`: For async test support

### Mocking Strategy

External services are mocked using pytest fixtures:
- The `mock_qa_chain` fixture simulates the QA chain responses
- All OpenAI and Qdrant calls are intercepted and replaced with test data
- Tests can run without internet connection or external service access
