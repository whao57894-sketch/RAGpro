import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable

from langchain_core.documents import Document

from src.vector_store import ChromaVectorStore


@dataclass(frozen=True)
class RetrievalResult:
    document: Document
    score: float
    source: str


def tokenize_for_retrieval(text: str) -> list[str]:
    """Tokenize Chinese-heavy enterprise documents for short-field retrieval."""

    normalized = normalize_query(text.lower())
    latin_tokens = re.findall(r"[a-zA-Z0-9]+", normalized)
    chinese_segments = re.findall(r"[\u4e00-\u9fff]+", normalized)

    tokens = list(latin_tokens)
    for segment in chinese_segments:
        tokens.extend(segment)
        for ngram_size in range(2, min(len(segment), 4) + 1):
            tokens.extend(
                segment[index : index + ngram_size]
                for index in range(len(segment) - ngram_size + 1)
            )
    return tokens


def normalize_query(text: str) -> str:
    text = text.strip()
    text = re.sub(r"[？?。！!；;，,]+$", "", text)
    return re.sub(r"\s+", " ", text)


def is_short_query(query: str) -> bool:
    normalized = normalize_query(query)
    compact = normalized.replace(" ", "")
    return len(compact) <= 8 or len(tokenize_for_retrieval(normalized)) <= 6


def expand_query(query: str) -> list[str]:
    normalized = normalize_query(query)
    focus = _extract_focus_phrase(normalized)
    candidates = [normalized]
    if focus and focus != normalized:
        candidates.append(focus)
    if focus and is_short_query(focus):
        candidates.extend(
            [
                f"{focus}：",
                f"{focus}是什么",
                f"文档中{focus}",
                f"{focus}字段",
                f"{focus}信息",
            ]
        )
    return list(dict.fromkeys(item for item in candidates if item))


class BM25Retriever:
    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents
        self.tokenized_documents = [tokenize_for_retrieval(_retrieval_text(document)) for document in documents]
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
            if score <= 0:
                continue
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

    def search(
        self,
        query: str,
        top_k: int = 4,
        bm25_top_k: int | None = None,
        vector_top_k: int | None = None,
    ) -> list[Document]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        expanded_queries = expand_query(query)
        bm25_top_k = bm25_top_k or max(top_k * 3, 12)
        vector_top_k = vector_top_k or max(top_k * 2, 8)
        combined_scores: dict[str, RetrievalResult] = {}

        for index, rewritten_query in enumerate(expanded_queries):
            query_weight = 1.0 if index == 0 else 0.82
            vector_weight, bm25_weight = self._weights_for_query(rewritten_query)
            vector_results = self.vector_store.similarity_search(rewritten_query, top_k=vector_top_k)
            bm25_results = self.bm25_retriever.search(rewritten_query, top_k=bm25_top_k)
            merged = self.merge_results(
                vector_results,
                bm25_results,
                vector_weight=vector_weight,
                bm25_weight=bm25_weight,
            )
            for result in merged:
                key = _document_key(result.document)
                weighted_score = result.score * query_weight
                existing = combined_scores.get(key)
                if existing:
                    combined_scores[key] = RetrievalResult(
                        document=_merge_document_metadata(existing.document, result.document),
                        score=existing.score + weighted_score,
                        source="hybrid",
                    )
                else:
                    combined_scores[key] = RetrievalResult(
                        document=result.document,
                        score=weighted_score,
                        source=result.source,
                    )

        ranked = sorted(combined_scores.values(), key=lambda item: item.score, reverse=True)
        return [result.document for result in ranked[:top_k]]

    def merge_results(
        self,
        vector_results: list[Document],
        bm25_results: list[RetrievalResult],
        *,
        vector_weight: float,
        bm25_weight: float,
    ) -> list[RetrievalResult]:
        combined: dict[str, RetrievalResult] = {}

        for rank, document in enumerate(vector_results, start=1):
            key = _document_key(document)
            score = 1.0 / rank
            combined[key] = RetrievalResult(
                document=_copy_document_with_metadata(document, {"source": "vector"}),
                score=score * vector_weight,
                source="vector",
            )

        for rank, result in enumerate(bm25_results, start=1):
            key = _document_key(result.document)
            score = 1.0 / rank
            existing = combined.get(key)
            if existing:
                combined[key] = RetrievalResult(
                    document=_merge_document_metadata(existing.document, result.document),
                    score=existing.score + score * bm25_weight,
                    source="hybrid",
                )
            else:
                combined[key] = RetrievalResult(
                    document=_copy_document_with_metadata(result.document, {"source": "bm25"}),
                    score=score * bm25_weight,
                    source="bm25",
                )

        return sorted(combined.values(), key=lambda item: item.score, reverse=True)

    def _weights_for_query(self, query: str) -> tuple[float, float]:
        if is_short_query(query):
            return 0.35, 0.65
        return self.vector_weight, self.bm25_weight


def _document_key(document: Document) -> str:
    metadata = document.metadata or {}
    file_name = metadata.get("file_name", "unknown")
    chunk_index = metadata.get("chunk_index", metadata.get("document_index", 0))
    return f"{file_name}:{chunk_index}:{document.page_content}"


def _merge_document_metadata(primary: Document, secondary: Document) -> Document:
    metadata = dict(secondary.metadata or {})
    metadata.update(primary.metadata or {})
    primary_sources = _metadata_sources(primary.metadata or {})
    secondary_sources = _metadata_sources(secondary.metadata or {})
    metadata["retrieval_sources"] = sorted(primary_sources | secondary_sources)
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


def _retrieval_text(document: Document) -> str:
    return str(document.metadata.get("retrieval_text") or document.page_content)


def _extract_focus_phrase(query: str) -> str:
    normalized = normalize_query(query)
    match = re.search(r"(.+?)(?:是什么|是啥|是多少|叫什么|叫啥|有哪些)$", normalized)
    focus = match.group(1) if match else normalized
    focus = re.sub(r"^(请问|麻烦问下|帮我看下|帮我看看|文档里|文档中|里面|这个|该|我的)", "", focus)
    return focus.strip()


def _copy_document_with_metadata(document: Document, extra_metadata: dict[str, object]) -> Document:
    metadata = dict(document.metadata or {})
    metadata.update(extra_metadata)
    return Document(page_content=document.page_content, metadata=metadata)


def _metadata_sources(metadata: dict) -> set[str]:
    sources = metadata.get("retrieval_sources")
    if isinstance(sources, list):
        return {str(item) for item in sources}
    if isinstance(sources, str) and sources:
        return {item.strip() for item in sources.split(",") if item.strip()}
    source = metadata.get("source")
    return {str(source)} if source else set()
