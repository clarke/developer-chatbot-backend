from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
import pytest
from app.main import app


client = TestClient(app)


@pytest.fixture
def mock_qa_chain():
    with patch("app.main.qa_chain") as mock:
        mock.return_value = {
            "result": "This is a test answer",
            "source_documents": [
                MagicMock(
                    metadata={"source": "test_file.py"},
                    page_content="This is a test document content"
                )
            ]
        }
        yield mock


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ask_endpoint_success(mock_qa_chain):
    test_question = "What is the meaning of life?"
    response = client.post("/ask", json={"question": test_question})

    assert response.status_code == 200
    assert response.json() == {
        "answer": "This is a test answer",
        "sources": [
            {
                "source": "test_file.py",
                "snippet": "This is a test document content"
            }
        ]
    }

    mock_qa_chain.assert_called_once_with(test_question)


def test_ask_endpoint_invalid_request():
    response = client.post("/ask", json={})
    assert response.status_code == 422  # Validation error


def test_ask_endpoint_error(mock_qa_chain):
    mock_qa_chain.side_effect = Exception("Test error")

    response = client.post("/ask", json={"question": "test"})
    assert response.status_code == 500
    assert "Test error" in response.json()["detail"]
