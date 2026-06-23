from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient

from api.main import app
from scripts.day3.create_sample_documents import main as create_sample_documents


def main() -> None:
    create_sample_documents()
    client = TestClient(app)

    health = client.get("/health")
    if health.status_code != 200:
        raise SystemExit(f"Health check failed: {health.text}")

    sample_path = Path("data/day3_samples/sample_faq.txt")
    with sample_path.open("rb") as file:
        response = client.post(
            "/documents/upload",
            files={"file": (sample_path.name, file, "text/plain")},
        )
    if response.status_code != 200:
        raise SystemExit(f"Upload failed: {response.status_code} {response.text}")

    upload_data = response.json()
    list_response = client.get("/documents")
    if list_response.status_code != 200:
        raise SystemExit(f"List failed: {list_response.status_code} {list_response.text}")

    documents = list_response.json()["documents"]
    if not documents:
        raise SystemExit("No uploaded documents returned by list API")

    print("Upload API OK")
    print(upload_data)
    print(list_response.json())


if __name__ == "__main__":
    main()
