from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.documents import Document

from src.embeddings import DeterministicEmbeddingModel
from src.llm import RuleBasedTestChatModel
from src.qa_engine import QAEngine
from src.vector_store import ChromaVectorStore


def main() -> None:
    store = ChromaVectorStore(
        embedding_model=DeterministicEmbeddingModel(),
        collection_name="day6_qa_validation",
    )
    store.add_documents(
        [
            Document(
                page_content="所有问答结果必须基于检索到的文本块，并标注文件名和出处。",
                metadata={"file_name": "answer_policy.txt", "chunk_index": 0},
            ),
            Document(
                page_content="向量数据库负责存储文本块 embedding，并按相似度返回 Top-K 结果。",
                metadata={"file_name": "vector_policy.txt", "chunk_index": 1},
            ),
        ]
    )

    engine = QAEngine(vector_store=store, chat_model=RuleBasedTestChatModel(), top_k=2)
    questions = [
        "回答为什么要标注出处？",
        "向量数据库负责什么？",
    ]

    output_lines = []
    for question in questions:
        result = engine.answer(question)
        output_lines.extend(
            [
                f"Q: {result.question}",
                f"A: {result.answer}",
                "Sources: " + ", ".join(source.file_name for source in result.sources),
                "",
            ]
        )
        print(output_lines[-4])
        print(output_lines[-3])
        print(output_lines[-2])
        print()

    output_path = Path("docs/day6/qa_engine_result.txt")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(output_lines), encoding="utf-8")
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
