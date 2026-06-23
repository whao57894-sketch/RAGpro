from fastapi.testclient import TestClient

from api.main import create_app
from src.embeddings import DeterministicEmbeddingModel
from src.llm import RuleBasedTestChatModel


def test_full_upload_ask_clear_flow():
    app = create_app(
        embedding_model=DeterministicEmbeddingModel(),
        chat_model=RuleBasedTestChatModel(),
        collection_name="test_day11_full_flow",
    )
    client = TestClient(app)

    assert client.post("/qa/ask", json={"question": "Why cite sources?"}).status_code == 400

    empty_upload = client.post(
        "/documents/upload",
        files={"file": ("empty.txt", b"", "text/plain")},
    )
    assert empty_upload.status_code == 400

    upload = client.post(
        "/documents/upload",
        files={
            "file": (
                "answer_policy.txt",
                b"RAG answers must cite source file names. Source citation helps employees verify every answer.",
                "text/plain",
            )
        },
    )
    assert upload.status_code == 200

    qa = client.post("/qa/ask", json={"question": "Why must RAG answers cite source file names?"})
    assert qa.status_code == 200
    assert qa.json()["sources"][0]["file_name"] == "answer_policy.txt"

    clear = client.delete("/documents/clear")
    assert clear.status_code == 200
    assert client.get("/documents").json()["total"] == 0
