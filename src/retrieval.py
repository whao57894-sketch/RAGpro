import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from langchain_core.documents import Document

from src.vector_store import ChromaVectorStore


@dataclass(frozen=True)
class RetrievalResult:
    document: Document
    score: float
    source: str


def tokenize_for_retrieval(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]{1,2}", text.lower())


class BM25Retriever:
    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents
        self.tokenized_documents = [tokenize_for_retrieval(document.page_content) for document in documents]
        self.doc_freqs = [Counter(tokens) for tokens in self.tokenized_documents]
        self.doc_lengths = [len(tokens) for tokens in self.tokenized_documents]
        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0
        self.document_frequency = defaultdict(int)
        for tokens in self.tokenized_documents:
            for token in set(tokens):
                self.document_frequency[token] += 1

    def search(self, query: str, top_k: int = 4) -> list[RetrievalResult]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")
        if not self.documents:
            return []

        query_tokens = tokenize_for_retrieval(query)
        scores = []
        for index, doc_freq in enumerate(self.doc_freqs):
            score = self._score_document(query_tokens, doc_freq, self.doc_lengths[index])
            scores.append(score)

        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)[:top_k]
        results: list[RetrievalResult] = []
        for index, score in ranked:
            results.append(RetrievalResult(document=self.documents[index], score=round(score, 4), source="bm25"))
        return results

    def _score_document(self, query_tokens: list[str], doc_freq: Counter, doc_length: int) -> float:
        if not query_tokens or doc_length == 0:
            return 0.0

        score = 0.0
        k1 = 1.5
        b = 0.75
        for token in query_tokens:
            df = self.document_frequency.get(token, 0)
            if df == 0:
                continue
            idf = math.log(1 + (len(self.documents) - df + 0.5) / (df + 0.5))
            term_freq = doc_freq.get(token, 0)
            denominator = term_freq + k1 * (1 - b + b * doc_length / max(self.avg_doc_length, 1e-9))
            score += idf * (term_freq * (k1 + 1)) / denominator
        return score


class HybridRetriever:
    def __init__(
        self,
        vector_store: ChromaVectorStore,
        bm25_documents: list[Document],
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5,
    ) -> None:
        self.vector_store = vector_store
        self.bm25_retriever = BM25Retriever(bm25_documents)
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight

    def search(self, query: str, top_k: int = 4, bm25_top_k: int | None = None) -> list[Document]:
        bm25_top_k = bm25_top_k or top_k * 2
        vector_results = self.vector_store.similarity_search(query, top_k=top_k)
        bm25_results = self.bm25_retriever.search(query, top_k=bm25_top_k)
        merged = self.merge_results(vector_results, bm25_results)
        return [result.document for result in merged[:top_k]]

    def merge_results(
        self,
        vector_results: list[Document],
        bm25_results: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        combined: dict[str, RetrievalResult] = {}

        for rank, document in enumerate(vector_results, start=1):
            key = _document_key(document)
            score = 1.0 / rank
            combined[key] = RetrievalResult(
                document=document,
                score=score * self.vector_weight,
                source="vector",
            )

        for rank, result in enumerate(bm25_results, start=1):
            key = _document_key(result.document)
            score = 1.0 / rank
            existing = combined.get(key)
            if existing:
                combined[key] = RetrievalResult(
                    document=_merge_document_metadata(existing.document, result.document),
                    score=existing.score + score * self.bm25_weight,
                    source="hybrid",
                )
            else:
                combined[key] = RetrievalResult(
                    document=result.document,
                    score=score * self.bm25_weight,
                    source="bm25",
                )

        return sorted(combined.values(), key=lambda item: item.score, reverse=True)


def _document_key(document: Document) -> str:
    metadata = document.metadata or {}
    file_name = metadata.get("file_name", "unknown")
    chunk_index = metadata.get("chunk_index", metadata.get("document_index", 0))
    return f"{file_name}:{chunk_index}:{document.page_content}"


def _merge_document_metadata(primary: Document, secondary: Document) -> Document:
    metadata = dict(secondary.metadata or {})
    metadata.update(primary.metadata or {})
    metadata["retrieval_sources"] = sorted({primary.metadata.get("retrieval_sources", primary.metadata.get("source", "vector")), secondary.metadata.get("source", "bm25")})
    return Document(page_content=primary.page_content, metadata=metadata)


def evaluate_retrieval(
    retriever,
    questions: Iterable[dict[str, str]],
    top_k: int = 3,
) -> dict[str, float | int]:
    total = 0
    hits = 0
    for item in questions:
        total += 1
        question = item["question"]
        expected_file = item["expected_file"]
        results = [_as_document(result) for result in retriever.search(question, top_k=top_k)]
        if any(result.metadata.get("file_name") == expected_file for result in results):
            hits += 1
    return {
        "total": total,
        "hits": hits,
        "accuracy": round(hits / total if total else 0, 3),
    }


def _as_document(result) -> Document:
    if isinstance(result, RetrievalResult):
        return result.document
    return result
