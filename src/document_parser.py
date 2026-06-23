from pathlib import Path
from typing import Iterable

from langchain_core.documents import Document
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


class UnsupportedDocumentTypeError(ValueError):
    """Raised when the parser receives an unsupported document type."""


def parse_document(file_path: str | Path) -> list[Document]:
    path = Path(file_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")
    if not path.is_file():
        raise ValueError(f"Document path is not a file: {path}")

    extension = path.suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise UnsupportedDocumentTypeError(f"Unsupported document type: {extension}")

    loader = _build_loader(path, extension)
    documents = loader.load()
    return _normalize_documents(documents, path, extension)


def parse_documents(file_paths: Iterable[str | Path]) -> list[Document]:
    parsed: list[Document] = []
    for file_path in file_paths:
        parsed.extend(parse_document(file_path))
    return parsed


def _build_loader(path: Path, extension: str):
    if extension == ".pdf":
        return PyPDFLoader(str(path))
    if extension == ".docx":
        return Docx2txtLoader(str(path))
    if extension == ".txt":
        return TextLoader(str(path), encoding="utf-8")
    raise UnsupportedDocumentTypeError(f"Unsupported document type: {extension}")


def _normalize_documents(documents: list[Document], path: Path, extension: str) -> list[Document]:
    normalized: list[Document] = []
    for index, document in enumerate(documents):
        metadata = {
            **document.metadata,
            "file_name": path.name,
            "file_path": str(path),
            "file_extension": extension,
            "document_index": index,
        }
        normalized.append(Document(page_content=document.page_content, metadata=metadata))
    return normalized
