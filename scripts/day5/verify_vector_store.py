from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.documents import Document

from src.embeddings import DeterministicEmbeddingModel
from src.vector_store import ChromaVectorStore


def main() -> None:
    documents = [
        Document(
            page_content="员工可以上传 PDF、DOCX 和 TXT 文档，系统会解析并建立知识库。",
            metadata={"file_name": "upload_policy.txt", "chunk_index": 0},
        ),
        Document(
            page_content="所有问答结果必须基于检索到的文本块，并标注文件名和出处。",
            metadata={"file_name": "answer_policy.txt", "chunk_index": 1},
        ),
        Document(
            page_content="向量数据库负责存储文本块 embedding，并按相似度返回 Top-K 结果。",
            metadata={"file_name": "vector_policy.txt", "chunk_index": 2},
        ),
    ]

    store = ChromaVectorStore(
        embedding_model=DeterministicEmbeddingModel(),
        collection_name="day5_vector_store_validation",
    )
    ids = store.add_documents(documents)
    results = store.similarity_search("向量数据库 Top-K 检索", top_k=2)

    if len(ids) != 3:
        raise SystemExit(f"Expected 3 inserted ids, got {ids}")
    if store.count() != 3:
        raise SystemExit(f"Expected collection count 3, got {store.count()}")
    if not results or "向量数据库" not in results[0].page_content:
        raise SystemExit(f"Unexpected retrieval results: {results}")

    output_path = Path("docs/day5/vector_store_result.txt")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(
            [
                f"inserted_ids={ids}",
                f"top_1={results[0].page_content}",
                f"top_1_metadata={results[0].metadata}",
            ]
        ),
        encoding="utf-8",
    )

    print("Vector store OK")
    print(f"Inserted: {len(ids)}")
    for index, document in enumerate(results, start=1):
        print(f"Top {index}: {document.page_content}")
        print(f"Metadata: {document.metadata}")


if __name__ == "__main__":
    main()
