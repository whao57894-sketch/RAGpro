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
        collection_name="day9_api_chain",
    )
    client = TestClient(app)

    upload_path = Path("data/day9_samples/answer_policy.txt")
    upload_path.parent.mkdir(parents=True, exist_ok=True)
    upload_path.write_text(
        "RAG answers must cite source file names. Source citation helps employees verify every answer.",
        encoding="utf-8",
    )
    with upload_path.open("rb") as file:
        upload_response = client.post(
            "/documents/upload",
            files={"file": (upload_path.name, file, "text/plain")},
        )
    if upload_response.status_code != 200:
        raise SystemExit(f"Upload failed: {upload_response.status_code} {upload_response.text}")

    ask_response = client.post("/qa/ask", json={"question": "Why must RAG answers cite source file names?"})
    if ask_response.status_code != 200:
        raise SystemExit(f"QA failed: {ask_response.status_code} {ask_response.text}")

    clear_response = client.delete("/documents/clear")
    if clear_response.status_code != 200:
        raise SystemExit(f"Clear failed: {clear_response.status_code} {clear_response.text}")

    print("QA API OK")
    print(upload_response.json())
    print(ask_response.json())
    print(clear_response.json())


if __name__ == "__main__":
    main()
