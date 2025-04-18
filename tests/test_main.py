from unittest.mock import MagicMock
from fastapi import FastAPI, HTTPException, Depends
from fastapi.testclient import TestClient
import pytest
import warnings
from pydantic import BaseModel

# Filter out all deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="langchain")
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module="_pytest.assertion.rewrite"
)


class QueryRequest(BaseModel):
    question: str


def get_qa_chain():
    return MagicMock()


def create_app():
    app = FastAPI(title="Codebase QA System")

    @app.post("/ask")
    async def ask_codebase(
        query: QueryRequest,
        qa_chain: MagicMock = Depends(get_qa_chain)
    ):
        try:
            result = qa_chain(query.question)
            return {
                "answer": result["result"],
                "sources": [
                    {
                        "source": doc.metadata.get("source"),
                        "snippet": doc.page_content[:200]
                    }
                    for doc in result["source_documents"]
                ]
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/health")
    def health_check():
        return {"status": "ok"}

    return app


@pytest.fixture
def mock_qa_chain():
    return MagicMock()


@pytest.fixture
def app(mock_qa_chain):
    app = create_app()
    app.dependency_overrides[get_qa_chain] = lambda: mock_qa_chain
    return app


@pytest.fixture
def client(app):
    return TestClient(app)


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ask_endpoint_success(client, mock_qa_chain):
    mock_result = {
        "result": "This is a test answer",
        "source_documents": [
            MagicMock(
                metadata={"source": "test.py"},
                page_content="This is a test document"
            )
        ]
    }
    mock_qa_chain.return_value = mock_result

    response = client.post(
        "/ask",
        json={"question": "What is this code doing?"}
    )

    assert response.status_code == 200
    assert "answer" in response.json()
    assert "sources" in response.json()
    assert len(response.json()["sources"]) == 1
    assert response.json()["answer"] == "This is a test answer"
    assert response.json()["sources"][0]["source"] == "test.py"


def test_ask_endpoint_invalid_request(client):
    response = client.post(
        "/ask",
        json={"invalid": "request"}
    )
    assert response.status_code == 422


def test_ask_endpoint_error(client, mock_qa_chain):
    mock_qa_chain.side_effect = Exception("Test error")

    response = client.post(
        "/ask",
        json={"question": "What is this code doing?"}
    )

    assert response.status_code == 500
    assert "detail" in response.json()
