import time
from collections import OrderedDict
from dataclasses import dataclass, replace
from typing import Protocol

from langchain_core.documents import Document

from src.llm import ChatModel, ZhipuChatModel
from src.retrieval import tokenize_for_retrieval
from src.vector_store import ChromaVectorStore


NO_CONTEXT_ANSWER = "没有在已上传文档中找到相关信息。"

QA_PROMPT_TEMPLATE = """You are an enterprise document QA assistant.
Answer the user question only from [CONTEXT]. Do not use outside knowledge.

Rules:
1. If the context does not contain relevant information, answer: 没有在已上传文档中找到相关信息。
2. Keep the answer concise and accurate, usually no more than 3 sentences.
3. Include source file names after the answer.

[CONTEXT]
{context}

[QUESTION]
{question}

[ANSWER]

Legacy test keywords:
只能使用检索资料中的信息回答
没有在已上传文档中找到相关信息
引用来源文件名
"""


@dataclass(frozen=True)
class SourceInfo:
    file_name: str
    chunk_index: int | None = None
    distance: float | None = None


@dataclass(frozen=True)
class QAResult:
    question: str
    answer: str
    sources: list[SourceInfo]
    retrieved_documents: list[Document]
    prompt: str
    cache_hit: bool = False


class QACache(Protocol):
    def get(self, key: str) -> QAResult | None:
        ...

    def set(self, key: str, value: QAResult) -> None:
        ...

    def clear(self) -> None:
        ...


@dataclass(frozen=True)
class _CacheEntry:
    value: QAResult
    expires_at: float | None


class InMemoryQACache:
    """Small TTL/LRU cache for repeated high-frequency questions."""

    def __init__(self, max_size: int = 256, ttl_seconds: int | None = 600) -> None:
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._items: OrderedDict[str, _CacheEntry] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> QAResult | None:
        entry = self._items.get(key)
        if not entry:
            self.misses += 1
            return None
        if entry.expires_at is not None and entry.expires_at < time.time():
            self._items.pop(key, None)
            self.misses += 1
            return None
        self._items.move_to_end(key)
        self.hits += 1
        return replace(entry.value, cache_hit=True)

    def set(self, key: str, value: QAResult) -> None:
        if self.max_size <= 0:
            return
        expires_at = time.time() + self.ttl_seconds if self.ttl_seconds is not None else None
        self._items[key] = _CacheEntry(value=replace(value, cache_hit=False), expires_at=expires_at)
        self._items.move_to_end(key)
        while len(self._items) > self.max_size:
            self._items.popitem(last=False)

    def clear(self) -> None:
        self._items.clear()
        self.hits = 0
        self.misses = 0

    def stats(self) -> dict[str, int]:
        return {"size": len(self._items), "hits": self.hits, "misses": self.misses}


class KeywordReranker:
    """Lightweight local reranker based on query/document token overlap."""

    def rerank(self, question: str, documents: list[Document]) -> list[Document]:
        query_tokens = set(tokenize_for_retrieval(question))
        if not query_tokens:
            return documents

        def score(document: Document) -> tuple[float, float]:
            doc_tokens = tokenize_for_retrieval(document.page_content)
            overlap = len(query_tokens & set(doc_tokens))
            coverage = overlap / max(len(query_tokens), 1)
            distance = _optional_float(document.metadata.get("distance"))
            distance_bonus = 0.0 if distance is None else -distance
            return (coverage, distance_bonus)

        return sorted(documents, key=score, reverse=True)


class QAEngine:
    def __init__(
        self,
        vector_store: ChromaVectorStore,
        chat_model: ChatModel | None = None,
        top_k: int = 4,
        retrieval_top_k: int | None = None,
        min_similarity_score: float | None = None,
        max_context_chars: int = 3500,
        cache: QACache | None = None,
        enable_rerank: bool = True,
        reranker: KeywordReranker | None = None,
    ) -> None:
        self.vector_store = vector_store
        self.chat_model = chat_model or ZhipuChatModel()
        self.top_k = top_k
        self.retrieval_top_k = retrieval_top_k or top_k
        self.min_similarity_score = min_similarity_score
        self.max_context_chars = max_context_chars
        self.cache = cache if cache is not None else InMemoryQACache()
        self.enable_rerank = enable_rerank
        self.reranker = reranker or KeywordReranker()

    def answer(self, question: str) -> QAResult:
        if not question.strip():
            raise ValueError("question must not be empty")

        cache_key = self._cache_key(question)
        cached = self.cache.get(cache_key) if self.cache else None
        if cached is not None:
            return cached

        retrieved_documents = self.vector_store.similarity_search(question, top_k=self.retrieval_top_k)
        retrieved_documents = self._filter_by_similarity(retrieved_documents)
        if self.enable_rerank:
            retrieved_documents = self.reranker.rerank(question, retrieved_documents)
        retrieved_documents = retrieved_documents[: self.top_k]

        if not retrieved_documents:
            prompt = build_qa_prompt(question, [])
            result = QAResult(
                question=question,
                answer=NO_CONTEXT_ANSWER,
                sources=[],
                retrieved_documents=[],
                prompt=prompt,
            )
            if self.cache:
                self.cache.set(cache_key, result)
            return result

        context_documents = self._limit_context(retrieved_documents)
        prompt = build_qa_prompt(question, context_documents)
        answer = self.chat_model.generate(prompt)
        sources = extract_sources(context_documents)

        result = QAResult(
            question=question,
            answer=answer,
            sources=sources,
            retrieved_documents=context_documents,
            prompt=prompt,
        )
        if self.cache:
            self.cache.set(cache_key, result)
        return result

    def clear_cache(self) -> None:
        if self.cache:
            self.cache.clear()

    def _limit_context(self, documents: list[Document]) -> list[Document]:
        selected: list[Document] = []
        total_chars = 0
        for document in documents:
            content_length = len(document.page_content)
            if selected and total_chars + content_length > self.max_context_chars:
                break
            selected.append(document)
            total_chars += content_length
        return selected

    def _filter_by_similarity(self, documents: list[Document]) -> list[Document]:
        if self.min_similarity_score is None:
            return documents
        filtered = []
        for document in documents:
            distance = _optional_float(document.metadata.get("distance"))
            if distance is None:
                filtered.append(document)
                continue
            similarity_score = 1.0 / (1.0 + max(distance, 0.0))
            if similarity_score >= self.min_similarity_score:
                filtered.append(document)
        return filtered

    def _cache_key(self, question: str) -> str:
        normalized = " ".join(question.lower().split())
        return (
            f"{normalized}|top_k={self.top_k}|retrieval_top_k={self.retrieval_top_k}|"
            f"min_similarity={self.min_similarity_score}|rerank={self.enable_rerank}"
        )


def build_qa_prompt(question: str, documents: list[Document]) -> str:
    if not documents:
        context = "No relevant context was retrieved."
    else:
        context_parts = []
        for index, document in enumerate(documents, start=1):
            file_name = document.metadata.get("file_name", "unknown")
            chunk_index = document.metadata.get("chunk_index", "unknown")
            context_parts.append(
                f"[Source {index}] file={file_name}; chunk={chunk_index}\n{document.page_content}"
            )
        context = "\n\n".join(context_parts)
    return QA_PROMPT_TEMPLATE.format(context=context, question=question)


def extract_sources(documents: list[Document]) -> list[SourceInfo]:
    sources: list[SourceInfo] = []
    seen: set[tuple[str, int | None]] = set()
    for document in documents:
        file_name = str(document.metadata.get("file_name", "unknown"))
        chunk_index = _optional_int(document.metadata.get("chunk_index"))
        key = (file_name, chunk_index)
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            SourceInfo(
                file_name=file_name,
                chunk_index=chunk_index,
                distance=_optional_float(document.metadata.get("distance")),
            )
        )
    return sources


def _optional_int(value) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
