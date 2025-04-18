# Developer Chatbot Backend

A powerful codebase question-answering system that enables natural language interaction with your codebase. This application uses LangChain, Qdrant, and OpenAI to provide intelligent responses to questions about your code.

## Overview

This application provides a developer-friendly interface to interact with codebases using natural language. It consists of:

1. A FastAPI-based REST API that handles codebase queries
2. A vector database (Qdrant) for efficient code search and retrieval
3. Utility scripts for code ingestion and inspection

The system uses OpenAI's embeddings and language models to understand and respond to questions about your codebase, making it easier to navigate and understand complex codebases.

## Features

- Natural language querying of codebases
- Support for multiple programming languages
- Source code chunking and semantic search
- REST API for easy integration
- Docker support for containerized deployment

## Prerequisites

- Python 3.8+
- Docker and Docker Compose
- OpenAI API key
- Git (for repository ingestion)

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
OPENAI_API_KEY=your_openai_api_key
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=codebase
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
├── app/                    # FastAPI application
│   ├── main.py            # Main application code
│   └── Dockerfile         # Docker configuration
├── scripts/               # Utility scripts
│   ├── ingest_local_repo.py  # Code ingestion script
│   ├── inspect_qdrant.py     # Database inspection
│   └── query.py              # Query interface
├── qdrant_data/           # Qdrant database storage
├── docker-compose.yml     # Docker Compose configuration
└── requirements.txt       # Python dependencies
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

[Add your license information here] 