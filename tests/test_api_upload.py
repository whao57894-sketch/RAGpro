from pathlib import Path

from fastapi.testclient import TestClient

from api.main import create_app
from src.embeddings import DeterministicEmbeddingModel
from src.llm import RuleBasedTestChatModel


def test_upload_txt_and_list_documents(tmp_path):
    app = create_app(
        embedding_model=DeterministicEmbeddingModel(),
        chat_model=RuleBasedTestChatModel(),
        collection_name="test_upload_txt_and_list_documents",
    )
    client = TestClient(app)
    file_path = tmp_path / "policy.txt"
    file_path.write_text(
        "Enterprise RAG upload API test.\nDocuments should be parsed, split and inserted into vector store.",
        encoding="utf-8",
    )

    with file_path.open("rb") as file:
        upload_response = client.post(
            "/documents/upload",
            files={"file": ("policy.txt", file, "text/plain")},
        )

    assert upload_response.status_code == 200
    payload = upload_response.json()
    assert payload["file_name"] == "policy.txt"
    assert payload["chunk_count"] >= 1
    assert payload["vector_count"] == payload["chunk_count"]

    list_response = client.get("/documents")
    assert list_response.status_code == 200
    documents = list_response.json()["documents"]
    assert len(documents) == 1
    assert documents[0]["file_name"] == "policy.txt"


def test_upload_rejects_unsupported_file(tmp_path):
    app = create_app(
        embedding_model=DeterministicEmbeddingModel(),
        chat_model=RuleBasedTestChatModel(),
        collection_name="test_upload_rejects_unsupported_file",
    )
    client = TestClient(app)
    file_path = tmp_path / "table.xlsx"
    file_path.write_text("unsupported", encoding="utf-8")

    with file_path.open("rb") as file:
        response = client.post(
            "/documents/upload",
            files={"file": ("table.xlsx", file, "application/octet-stream")},
        )

    assert response.status_code == 400


def test_ask_requires_non_empty_knowledge_base():
    app = create_app(
        embedding_model=DeterministicEmbeddingModel(),
        chat_model=RuleBasedTestChatModel(),
        collection_name="test_ask_requires_non_empty_knowledge_base",
    )
    client = TestClient(app)

    response = client.post("/qa/ask", json={"question": "What is the answer?"})

    assert response.status_code == 400
    assert "Knowledge base is empty" in response.json()["detail"]


def test_full_upload_then_ask_chain(tmp_path):
    app = create_app(
        embedding_model=DeterministicEmbeddingModel(),
        chat_model=RuleBasedTestChatModel(),
        collection_name="test_full_upload_then_ask_chain",
    )
    client = TestClient(app)
    file_path = tmp_path / "answer_policy.txt"
    file_path.write_text(
        "RAG answers must cite source file names.\nSource citation helps employees verify every answer.",
        encoding="utf-8",
    )

    with file_path.open("rb") as file:
        upload_response = client.post(
            "/documents/upload",
            files={"file": ("answer_policy.txt", file, "text/plain")},
        )
    assert upload_response.status_code == 200

    ask_response = client.post("/qa/ask", json={"question": "Why must RAG answers cite source file names?"})
    assert ask_response.status_code == 200
    payload = ask_response.json()
    assert "cite source file names" in payload["answer"].lower()
    assert payload["sources"]
    assert payload["sources"][0]["file_name"] == "answer_policy.txt"


def test_clear_documents_endpoint(tmp_path):
    app = create_app(
        embedding_model=DeterministicEmbeddingModel(),
        chat_model=RuleBasedTestChatModel(),
        collection_name="test_clear_documents_endpoint",
    )
    client = TestClient(app)
    file_path = tmp_path / "vector_policy.txt"
    file_path.write_text(
        "The vector database stores chunk embeddings and returns Top-K similar chunks.",
        encoding="utf-8",
    )

    with file_path.open("rb") as file:
        upload_response = client.post(
            "/documents/upload",
            files={"file": ("vector_policy.txt", file, "text/plain")},
        )
    assert upload_response.status_code == 200

    clear_response = client.delete("/documents/clear")
    assert clear_response.status_code == 200
    assert clear_response.json()["cleared_documents"] == 1
    assert client.get("/documents").json()["total"] == 0
