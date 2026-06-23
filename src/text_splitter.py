from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


CHINESE_SEPARATORS = [
    "\n\n",
    "\n",
    "。",
    "！",
    "？",
    "；",
    ";",
    "，",
    ",",
    " ",
    "",
]

DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 80
DOCX_PARAGRAPH_CHUNK_SIZE = 280
DOCX_PARAGRAPH_CHUNK_OVERLAP = 40
STRUCTURED_SECTION_TYPES = {"heading", "table_row", "field_block"}


@dataclass(frozen=True)
class SplitConfig:
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP


def split_documents(
    documents: list[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    _validate_split_params(chunk_size, chunk_overlap)
    chunks: list[Document] = []
    for document in documents:
        if _should_preserve_as_single_chunk(document, chunk_size):
            chunks.append(document)
            continue
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=_resolve_chunk_size(document, chunk_size),
            chunk_overlap=_resolve_chunk_overlap(document, chunk_overlap),
            separators=CHINESE_SEPARATORS,
            keep_separator=True,
            add_start_index=True,
        )
        chunks.extend(splitter.split_documents([document]))
    return _normalize_chunk_metadata(chunks, chunk_size, chunk_overlap)


def compare_split_configs(
    documents: list[Document],
    configs: list[SplitConfig],
) -> list[dict[str, float | int]]:
    results: list[dict[str, float | int]] = []
    for config in configs:
        chunks = split_documents(
            documents,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )
        lengths = [len(chunk.page_content) for chunk in chunks]
        results.append(
            {
                "chunk_size": config.chunk_size,
                "chunk_overlap": config.chunk_overlap,
                "chunk_count": len(chunks),
                "min_length": min(lengths) if lengths else 0,
                "max_length": max(lengths) if lengths else 0,
                "avg_length": round(sum(lengths) / len(lengths), 2) if lengths else 0,
                "quality_score": _estimate_quality_score(chunks, config),
            }
        )
    return results


def _normalize_chunk_metadata(
    chunks: list[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    normalized: list[Document] = []
    for chunk_index, chunk in enumerate(chunks):
        resolved_chunk_size = _resolve_chunk_size(chunk, chunk_size)
        resolved_chunk_overlap = _resolve_chunk_overlap(chunk, chunk_overlap)
        metadata = {
            **chunk.metadata,
            "chunk_index": chunk_index,
            "chunk_size": resolved_chunk_size,
            "chunk_overlap": resolved_chunk_overlap,
            "chunk_length": len(chunk.page_content),
            "chunk_type": chunk.metadata.get("section_type", "paragraph"),
            "is_structured_chunk": _is_structured_document(chunk),
        }
        normalized.append(Document(page_content=chunk.page_content, metadata=metadata))
    return normalized


def _estimate_quality_score(chunks: list[Document], config: SplitConfig) -> float:
    if not chunks:
        return 0

    lengths = [len(chunk.page_content.strip()) for chunk in chunks]
    too_short_count = sum(length < config.chunk_size * 0.25 for length in lengths)
    too_long_count = sum(length > config.chunk_size for length in lengths)
    sentence_end_count = sum(
        chunk.page_content.strip().endswith(("。", "！", "？", ".", "!", "?"))
        for chunk in chunks
    )

    short_penalty = too_short_count / len(chunks)
    long_penalty = too_long_count / len(chunks)
    sentence_end_bonus = sentence_end_count / len(chunks)

    score = 100 - short_penalty * 25 - long_penalty * 35 + sentence_end_bonus * 10
    return round(max(0, min(100, score)), 2)


def _validate_split_params(chunk_size: int, chunk_overlap: int) -> None:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must not be negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")


def _should_preserve_as_single_chunk(document: Document, chunk_size: int) -> bool:
    if _is_structured_document(document):
        return True
    if len(document.page_content) <= min(chunk_size, DOCX_PARAGRAPH_CHUNK_SIZE) and document.metadata.get("file_extension") == ".docx":
        return True
    return False


def _resolve_chunk_size(document: Document, default_chunk_size: int) -> int:
    if document.metadata.get("file_extension") == ".docx" and not _is_structured_document(document):
        return min(default_chunk_size, DOCX_PARAGRAPH_CHUNK_SIZE)
    return default_chunk_size


def _resolve_chunk_overlap(document: Document, default_chunk_overlap: int) -> int:
    if document.metadata.get("file_extension") == ".docx" and not _is_structured_document(document):
        return min(default_chunk_overlap, DOCX_PARAGRAPH_CHUNK_OVERLAP)
    return default_chunk_overlap


def _is_structured_document(document: Document) -> bool:
    return document.metadata.get("section_type") in STRUCTURED_SECTION_TYPES
