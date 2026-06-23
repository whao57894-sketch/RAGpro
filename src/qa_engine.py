import time
from collections import OrderedDict
from dataclasses import dataclass, replace
from typing import Callable, Protocol

from langchain_core.documents import Document

from src.llm import ChatModel, ZhipuChatModel
from src.retrieval import HybridRetriever, is_short_query, tokenize_for_retrieval
from src.vector_store import ChromaVectorStore


NO_CONTEXT_ANSWER = "没有在已上传文档中找到相关信息。"

QA_PROMPT_TEMPLATE = """You are an enterprise document QA assistant.
Answer the user question only from [CONTEXT]. Do not use outside knowledge.

Rules:
1. If the context does not contain relevant information, answer: 没有在已上传文档中找到相关信息。
2. Keep the answer concise and accurate, usually no more than 3 sentences.
3. Include source file names after the answer.
4. For field/value questions, prefer the exact value from the most relevant title, paragraph, or table row.

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

        compact_question = _compact_text(question)

        def score(document: Document) -> tuple[float, float, float]:
            text = _retrieval_text(document)
            doc_tokens = set(tokenize_for_retrieval(text))
            overlap = len(query_tokens & set(doc_tokens))
            coverage = overlap / max(len(query_tokens), 1)
            distance = _optional_float(document.metadata.get("distance"))
            distance_bonus = 0.0 if distance is None else -distance
            compact_text = _compact_text(text)
            exact_phrase_bonus = 1.0 if compact_question and compact_question in compact_text else 0.0
            heading_bonus = 0.2 if _has_overlap(question, str(document.metadata.get("heading_path", ""))) else 0.0
            structured_bonus = 0.25 if document.metadata.get("section_type") in {"table_row", "field_block", "heading"} else 0.0
            colon_bonus = 0.15 if is_short_query(question) and ("：" in text or ":" in text) else 0.0
            field_key_bonus = 0.25 if _matches_field_keys(question, document) else 0.0
            total_score = coverage * 2.0 + exact_phrase_bonus + heading_bonus + structured_bonus + colon_bonus + field_key_bonus
            return (total_score, coverage, distance_bonus)

        scored_documents = []
        for document in documents:
            total_score, coverage, distance_bonus = score(document)
            scored_documents.append(
                _copy_document_with_metadata(
                    document,
                    {
                        "rerank_score": round(total_score, 6),
                        "query_coverage": round(coverage, 6),
                        "distance_bonus": round(distance_bonus, 6),
                    },
                )
            )

        return sorted(
            scored_documents,
            key=lambda document: (
                float(document.metadata.get("rerank_score", 0.0)),
                float(document.metadata.get("query_coverage", 0.0)),
                float(document.metadata.get("distance_bonus", 0.0)),
            ),
            reverse=True,
        )


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
        keyword_documents_provider: Callable[[], list[Document]] | None = None,
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
        self.keyword_documents_provider = keyword_documents_provider

    def answer(self, question: str) -> QAResult:
        if not question.strip():
            raise ValueError("question must not be empty")

        cache_key = self._cache_key(question)
        cached = self.cache.get(cache_key) if self.cache else None
        if cached is not None:
            return cached

        retrieved_documents = self._retrieve_documents(question)
        retrieved_documents = self._filter_by_similarity(question, retrieved_documents)
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
        supporting_documents = [] if _is_no_context_answer(answer) else select_supporting_documents(question, answer, context_documents)
        sources = extract_sources(supporting_documents)

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

    def _retrieve_documents(self, question: str) -> list[Document]:
        if self.keyword_documents_provider is not None:
            keyword_documents = self._keyword_search(question)
            if keyword_documents:
                return keyword_documents
        return self.vector_store.similarity_search(question, top_k=self._candidate_retrieval_top_k(question))

    def _keyword_search(self, question: str) -> list[Document]:
        if self.keyword_documents_provider is None:
            return []
        documents = self.keyword_documents_provider()
        if not documents:
            return []
        retriever = HybridRetriever(
            vector_store=self.vector_store,
            bm25_documents=documents,
            vector_weight=0.45,
            bm25_weight=0.55,
        )
        return retriever.search(
            question,
            top_k=self._candidate_retrieval_top_k(question),
            bm25_top_k=max(self._candidate_retrieval_top_k(question) * 2, self.top_k + 8),
            vector_top_k=max(self._candidate_retrieval_top_k(question), self.top_k + 4),
        )

    def _filter_by_similarity(self, question: str, documents: list[Document]) -> list[Document]:
        if is_short_query(question):
            return documents
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

    def _candidate_retrieval_top_k(self, question: str) -> int:
        if is_short_query(question):
            return max(self.retrieval_top_k, self.top_k + 8)
        return max(self.retrieval_top_k, self.top_k + 4)

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


def select_supporting_documents(question: str, answer: str, documents: list[Document], max_sources: int = 2) -> list[Document]:
    if not documents:
        return []

    scored: list[tuple[float, Document]] = []
    for document in documents:
        score = _support_score(question, answer, document)
        if score > 0:
            scored.append((score, document))

    if not scored:
        return documents[:1]

    scored.sort(key=lambda item: item[0], reverse=True)
    best_score = scored[0][0]
    threshold = max(best_score * 0.75, best_score - 1.5)
    selected = [document for score, document in scored if score >= threshold]
    return selected[:max_sources]


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


def _is_no_context_answer(answer: str) -> bool:
    normalized = answer.strip().lower()
    no_context_markers = [
        "没有在已上传文档中找到相关信息",
        "no relevant information",
        "no relevant context",
    ]
    return any(marker in normalized for marker in no_context_markers)


def _support_score(question: str, answer: str, document: Document) -> float:
    question_tokens = set(tokenize_for_retrieval(question))
    answer_tokens = set(tokenize_for_retrieval(answer))
    text = _retrieval_text(document)
    doc_tokens = set(tokenize_for_retrieval(text))
    question_overlap = len(question_tokens & doc_tokens)
    answer_overlap = len(answer_tokens & doc_tokens)
    exact_phrase_bonus = 1.0 if _compact_text(question) and _compact_text(question) in _compact_text(text) else 0.0
    structured_bonus = 0.35 if document.metadata.get("section_type") in {"table_row", "field_block"} else 0.0
    return question_overlap * 2.0 + answer_overlap * 0.8 + exact_phrase_bonus + structured_bonus


def _matches_field_keys(question: str, document: Document) -> bool:
    field_keys = str(document.metadata.get("field_keys", ""))
    if not field_keys:
        return False
    compact_question = _compact_text(question)
    return any(
        key and (_compact_text(key) in compact_question or compact_question in _compact_text(key))
        for key in field_keys.split("|")
    )


def _has_overlap(text: str, candidate: str) -> bool:
    return bool(set(tokenize_for_retrieval(text)) & set(tokenize_for_retrieval(candidate)))


def _retrieval_text(document: Document) -> str:
    return str(document.metadata.get("retrieval_text") or document.page_content)


def _compact_text(text: str) -> str:
    return "".join(text.split())


def _copy_document_with_metadata(document: Document, extra_metadata: dict[str, object]) -> Document:
    metadata = dict(document.metadata or {})
    metadata.update(extra_metadata)
    return Document(page_content=document.page_content, metadata=metadata)
