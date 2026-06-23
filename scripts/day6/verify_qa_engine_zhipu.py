from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.documents import Document

from src.embeddings import DeterministicEmbeddingModel
from src.llm import ZhipuChatModel
from src.qa_engine import QAEngine
from src.vector_store import ChromaVectorStore


def main() -> None:
    store = ChromaVectorStore(
        embedding_model=DeterministicEmbeddingModel(),
        collection_name="day6_qa_zhipu_validation",
    )
    store.add_documents(
        [
            Document(
                page_content="企业文档问答系统的回答必须基于检索到的资料，不能编造未在文档中出现的信息。",
                metadata={"file_name": "qa_policy.txt", "chunk_index": 0},
            ),
            Document(
                page_content="如果检索资料中没有相关内容，系统应明确告知没有在已上传文档中找到相关信息。",
                metadata={"file_name": "fallback_policy.txt", "chunk_index": 1},
            ),
        ]
    )

    engine = QAEngine(vector_store=store, chat_model=ZhipuChatModel(), top_k=2)
    result = engine.answer("如果文档里没有相关内容，系统应该怎么回答？")

    print(f"Q: {result.question}")
    print(f"A: {result.answer}")
    print("Sources: " + ", ".join(source.file_name for source in result.sources))


if __name__ == "__main__":
    main()
