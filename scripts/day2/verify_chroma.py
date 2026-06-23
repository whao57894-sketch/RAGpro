from pathlib import Path


def main() -> None:
    try:
        import chromadb
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "ChromaDB is not installed. Current Python 3.13 environment failed to install "
            "chromadb==0.6.3 because chroma-hnswlib requires Microsoft C++ Build Tools. "
            "Recommended fix: use Python 3.11/3.12, or install Microsoft C++ Build Tools and retry."
        ) from exc

    client = chromadb.Client()
    collection = client.get_or_create_collection(name="day2_validation")
    collection.upsert(
        ids=["policy-1", "policy-2"],
        documents=[
            "Employees can ask questions based on uploaded enterprise documents.",
            "Answers must cite document sources and avoid unsupported claims.",
        ],
        embeddings=[
            [0.1, 0.2, 0.3, 0.4],
            [0.9, 0.1, 0.1, 0.1],
        ],
        metadatas=[
            {"source": "enterprise_policy_test.pdf", "page": 1},
            {"source": "enterprise_policy_test.pdf", "page": 1},
        ],
    )
    result = collection.query(query_embeddings=[[0.9, 0.1, 0.1, 0.1]], n_results=1)
    document = result["documents"][0][0]
    metadata = result["metadatas"][0][0]

    if "cite document sources" not in document:
        raise SystemExit(f"Unexpected Chroma result: {result}")

    output_path = Path("docs/day2/chroma_result.txt")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(f"document={document}\nmetadata={metadata}\n", encoding="utf-8")
    print("ChromaDB OK")
    print(document)
    print(metadata)


if __name__ == "__main__":
    main()
