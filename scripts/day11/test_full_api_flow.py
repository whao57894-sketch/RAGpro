from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient

from api.main import create_app
from src.embeddings import DeterministicEmbeddingModel
from src.llm import RuleBasedTestChatModel


def main() -> None:
    app = create_app(
        embedding_model=DeterministicEmbeddingModel(),
        chat_model=RuleBasedTestChatModel(),
        collection_name="day11_full_flow",
    )
    client = TestClient(app)

    empty_qa = client.post("/qa/ask", json={"question": "Why cite sources?"})
    assert empty_qa.status_code == 400

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

    docs = client.get("/documents")
    assert docs.status_code == 200
    assert docs.json()["total"] == 1

    qa = client.post("/qa/ask", json={"question": "Why must RAG answers cite source file names?"})
    assert qa.status_code == 200
    payload = qa.json()
    assert "cite source file names" in payload["answer"].lower()
    assert payload["sources"][0]["file_name"] == "answer_policy.txt"

    clear = client.delete("/documents/clear")
    assert clear.status_code == 200

    print("Day 11 full API flow OK")
    print(payload)


if __name__ == "__main__":
    main()
