import json
import sys
import time
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.embeddings import DeterministicEmbeddingModel
from src.llm import ChatModel
from src.qa_engine import InMemoryQACache, QAEngine
from src.vector_store import ChromaVectorStore


REPORT_DIR = Path("day13_docs")
REPORT_PATH = REPORT_DIR / "performance_optimization_report.md"
RESULTS_PATH = REPORT_DIR / "performance_optimization_results.json"


class SlowCountingChatModel(ChatModel):
    def __init__(self, delay_seconds: float = 0.05) -> None:
        self.delay_seconds = delay_seconds
        self.calls = 0

    def generate(self, prompt: str) -> str:
        self.calls += 1
        time.sleep(self.delay_seconds)
        return "Use source citations and answer concisely based on retrieved context."


def build_documents() -> list[Document]:
    return [
        Document(page_content="Annual leave: full-time employees receive 15 annual leave days each year.", metadata={"file_name": "leave_policy.txt", "chunk_index": 0}),
        Document(page_content="Expense policy: taxi receipts must be submitted within 30 days.", metadata={"file_name": "expense_policy.txt", "chunk_index": 0}),
        Document(page_content="VPN FAQ: failed VPN login requires resetting the MFA token.", metadata={"file_name": "vpn_faq.txt", "chunk_index": 0}),
        Document(page_content="Support FAQ: P1 tickets require first response within 15 minutes.", metadata={"file_name": "support_faq.txt", "chunk_index": 0}),
        Document(page_content="SLA policy: standard incidents target resolution is 2 business days.", metadata={"file_name": "sla_policy.txt", "chunk_index": 0}),
    ]


def build_questions() -> list[dict[str, str]]:
    return [
        {"question": "How many annual leave days are provided?", "expected_file": "leave_policy.txt"},
        {"question": "When must taxi receipts be submitted?", "expected_file": "expense_policy.txt"},
        {"question": "What should be reset when VPN login fails?", "expected_file": "vpn_faq.txt"},
        {"question": "What is the P1 first response target?", "expected_file": "support_faq.txt"},
        {"question": "What is the standard incident target resolution?", "expected_file": "sla_policy.txt"},
    ]


def run_benchmark() -> dict[str, Any]:
    store = ChromaVectorStore(
        embedding_model=DeterministicEmbeddingModel(dimensions=128),
        collection_name=f"day13_perf_{int(time.time() * 1000)}",
    )
    store.add_documents(build_documents())

    chat_model = SlowCountingChatModel()
    cache = InMemoryQACache(max_size=64, ttl_seconds=300)
    engine = QAEngine(
        vector_store=store,
        chat_model=chat_model,
        top_k=3,
        retrieval_top_k=5,
        min_similarity_score=0.05,
        cache=cache,
        enable_rerank=True,
    )

    question = "How many annual leave days are provided?"
    first_start = time.perf_counter()
    first = engine.answer(question)
    first_ms = round((time.perf_counter() - first_start) * 1000, 2)

    second_start = time.perf_counter()
    second = engine.answer(question)
    second_ms = round((time.perf_counter() - second_start) * 1000, 2)

    config_results = []
    for top_k, retrieval_top_k, min_similarity, enable_rerank in [
        (2, 2, None, False),
        (3, 5, None, True),
        (3, 5, 0.05, True),
        (4, 5, 0.05, True),
    ]:
        eval_engine = QAEngine(
            vector_store=store,
            chat_model=SlowCountingChatModel(delay_seconds=0),
            top_k=top_k,
            retrieval_top_k=retrieval_top_k,
            min_similarity_score=min_similarity,
            cache=InMemoryQACache(max_size=0),
            enable_rerank=enable_rerank,
        )
        hits = 0
        started = time.perf_counter()
        for item in build_questions():
            result = eval_engine.answer(item["question"])
            if any(source.file_name == item["expected_file"] for source in result.sources):
                hits += 1
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        config_results.append(
            {
                "top_k": top_k,
                "retrieval_top_k": retrieval_top_k,
                "min_similarity_score": min_similarity,
                "rerank": enable_rerank,
                "accuracy": round(hits / len(build_questions()), 4),
                "elapsed_ms": elapsed_ms,
            }
        )

    results = {
        "cache": {
            "first_ms": first_ms,
            "second_ms": second_ms,
            "speedup": round(first_ms / max(second_ms, 0.01), 2),
            "first_cache_hit": first.cache_hit,
            "second_cache_hit": second.cache_hit,
            "llm_calls": chat_model.calls,
            "stats": cache.stats(),
        },
        "retrieval_configs": config_results,
        "recommended_config": max(config_results, key=lambda item: (item["accuracy"], -item["elapsed_ms"])),
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    REPORT_PATH.write_text(_render_report(results), encoding="utf-8")
    return results


def _render_report(results: dict[str, Any]) -> str:
    cache = results["cache"]
    recommended = results["recommended_config"]
    lines = [
        "# Day 13 Performance Optimization Report",
        "",
        "## Cache Result",
        "",
        f"- First request: {cache['first_ms']} ms",
        f"- Repeated request: {cache['second_ms']} ms",
        f"- Speedup: {cache['speedup']}x",
        f"- LLM calls: {cache['llm_calls']}",
        f"- Cache stats: {cache['stats']}",
        "",
        "## Retrieval Parameter Comparison",
        "",
        "| Top-K | Retrieval Top-K | Min similarity | Rerank | Accuracy | Elapsed ms |",
        "| ---: | ---: | --- | --- | ---: | ---: |",
    ]
    for item in results["retrieval_configs"]:
        lines.append(
            f"| {item['top_k']} | {item['retrieval_top_k']} | {item['min_similarity_score']} | {item['rerank']} | {item['accuracy']:.2%} | {item['elapsed_ms']} |"
        )
    lines.extend(
        [
            "",
            "## Recommended Configuration",
            "",
            f"- Top-K: {recommended['top_k']}",
            f"- Retrieval Top-K: {recommended['retrieval_top_k']}",
            f"- Min similarity score: {recommended['min_similarity_score']}",
            f"- Rerank enabled: {recommended['rerank']}",
            "",
            "## Notes",
            "",
            "- In-memory TTL/LRU cache is enough for local development and single-process deployment.",
            "- Redis can replace `InMemoryQACache` later when multiple backend workers share cache state.",
            "- Prompt text was shortened to enforce concise context-only answers.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    results = run_benchmark()
    cache = results["cache"]
    print(f"First request: {cache['first_ms']} ms")
    print(f"Repeated request: {cache['second_ms']} ms")
    print(f"Speedup: {cache['speedup']}x")
    print(f"LLM calls: {cache['llm_calls']}")
    print(f"Report: {REPORT_PATH}")
    print(f"Raw results: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
