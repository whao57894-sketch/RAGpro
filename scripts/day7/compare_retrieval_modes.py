from pathlib import Path
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.document_parser import parse_documents
from src.embeddings import DeterministicEmbeddingModel
from src.retrieval import BM25Retriever, HybridRetriever, evaluate_retrieval
from src.text_splitter import split_documents
from src.vector_store import ChromaVectorStore


def main() -> None:
    eval_set_path = Path("docs/day7/eval_set.json")
    eval_items = json.loads(eval_set_path.read_text(encoding="utf-8"))

    sample_paths = sorted(Path("data/day7_corpus").glob("*.txt"))
    documents = []
    for path in sample_paths:
        if path.suffix.lower() in {".pdf", ".docx", ".txt"} and path.exists():
            documents.extend(parse_documents([path]))
        elif path.exists():
            documents.append(
                __import__("langchain_core.documents", fromlist=["Document"]).Document(
                    page_content=path.read_text(encoding="utf-8"),
                    metadata={"file_name": path.name, "file_path": str(path), "chunk_index": 0},
                )
            )

    chunks = split_documents(documents, chunk_size=500, chunk_overlap=80)
    store = ChromaVectorStore(
        embedding_model=DeterministicEmbeddingModel(),
        collection_name="day7_retrieval_eval",
    )
    store.add_documents(chunks)

    bm25_retriever = BM25Retriever(chunks)
    hybrid_retriever = HybridRetriever(store, chunks)

    vector_metrics = evaluate_retrieval(
        type("VectorOnly", (), {"search": lambda self, q, top_k=3: store.similarity_search(q, top_k=top_k)})(),
        eval_items,
    )
    bm25_metrics = evaluate_retrieval(bm25_retriever, eval_items)
    hybrid_metrics = evaluate_retrieval(hybrid_retriever, eval_items)

    output_lines = [
        "# Day 7 检索效果对比",
        "",
        "| Mode | Total | Hits | Accuracy |",
        "| --- | --- | --- | --- |",
        f"| Pure Vector | {vector_metrics['total']} | {vector_metrics['hits']} | {vector_metrics['accuracy']} |",
        f"| BM25 Only | {bm25_metrics['total']} | {bm25_metrics['hits']} | {bm25_metrics['accuracy']} |",
        f"| Hybrid | {hybrid_metrics['total']} | {hybrid_metrics['hits']} | {hybrid_metrics['accuracy']} |",
        "",
        "## Sample Queries",
        "",
    ]

    for item in eval_items:
        question = item["question"]
        vector_top = store.similarity_search(question, top_k=3)
        hybrid_top = hybrid_retriever.search(question, top_k=3)
        output_lines.extend(
            [
                f"### {question}",
                "",
                f"- Expected: {item['expected_file']}",
                f"- Vector: {', '.join(doc.metadata.get('file_name', 'unknown') for doc in vector_top)}",
                f"- Hybrid: {', '.join(doc.metadata.get('file_name', 'unknown') for doc in hybrid_top)}",
                "",
            ]
        )

    output_path = Path("docs/day7/retrieval_comparison.md")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(output_lines), encoding="utf-8")

    print("\n".join(output_lines))
    print(f"\nSaved {output_path}")


if __name__ == "__main__":
    main()
