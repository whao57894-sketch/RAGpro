from langchain_core.documents import Document

from scripts.day13.benchmark_performance import run_benchmark
from src.embeddings import DeterministicEmbeddingModel
from src.llm import ChatModel
from src.qa_engine import InMemoryQACache, QAEngine
from src.vector_store import ChromaVectorStore


class CountingChatModel(ChatModel):
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, prompt: str) -> str:
        self.calls += 1
        return "cached answer"


def test_qa_engine_uses_cache_for_repeated_questions():
    store = ChromaVectorStore(
        embedding_model=DeterministicEmbeddingModel(),
        collection_name="test_day13_cache",
    )
    store.add_documents([Document(page_content="Annual leave is 15 days.", metadata={"file_name": "leave.txt", "chunk_index": 0})])
    chat_model = CountingChatModel()
    engine = QAEngine(
        vector_store=store,
        chat_model=chat_model,
        cache=InMemoryQACache(max_size=8, ttl_seconds=60),
    )

    first = engine.answer("How many annual leave days?")
    second = engine.answer("How many annual leave days?")

    assert first.cache_hit is False
    assert second.cache_hit is True
    assert chat_model.calls == 1


def test_day13_benchmark_generates_metrics():
    results = run_benchmark()

    assert results["cache"]["second_cache_hit"] is True
    assert results["cache"]["llm_calls"] == 1
    assert results["cache"]["speedup"] > 1
    assert results["recommended_config"]["accuracy"] >= 0.8
