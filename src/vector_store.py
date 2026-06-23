from pathlib import Path
from typing import Any
from uuid import uuid4

import chromadb
from chromadb.config import Settings
from chromadb.api.models.Collection import Collection
from langchain_core.documents import Document

from src.embeddings import EmbeddingModel, ZhipuEmbeddingModel


DEFAULT_COLLECTION_NAME = "enterprise_documents"


class ChromaVectorStore:
    def __init__(
        self,
        embedding_model: EmbeddingModel | None = None,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        persist_directory: str | Path | None = None,
    ) -> None:
        self.embedding_model = embedding_model or ZhipuEmbeddingModel()
        self.collection_name = collection_name
        self.persist_directory = Path(persist_directory) if persist_directory else None
        self.client = self._create_client()
        self.collection = self._get_collection(collection_name)

    def add_documents(self, documents: list[Document]) -> list[str]:
        if not documents:
            return []

        ids = [self._document_id(document, index) for index, document in enumerate(documents)]
        texts = [document.page_content for document in documents]
        embeddings = self.embedding_model.embed_documents(texts)
        metadatas = [self._sanitize_metadata(document.metadata) for document in documents]

        self.collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        return ids

    def similarity_search(self, query: str, top_k: int = 4) -> list[Document]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        query_embedding = self.embedding_model.embed_query(query)
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        return self._query_result_to_documents(result)

    def count(self) -> int:
        return self.collection.count()

    def _create_client(self):
        if self.persist_directory:
            self.persist_directory.mkdir(parents=True, exist_ok=True)
            return chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=Settings(anonymized_telemetry=False),
            )
        return chromadb.Client(settings=Settings(anonymized_telemetry=False))

    def _get_collection(self, collection_name: str) -> Collection:
        return self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Enterprise document chunks"},
        )

    @staticmethod
    def _document_id(document: Document, index: int) -> str:
        metadata = document.metadata
        file_name = metadata.get("file_name", "document")
        chunk_index = metadata.get("chunk_index", index)
        source_index = metadata.get("document_index", 0)
        return f"{file_name}-{source_index}-{chunk_index}-{uuid4().hex[:8]}"

    @staticmethod
    def _sanitize_metadata(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
        sanitized: dict[str, str | int | float | bool] = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                sanitized[key] = value
            elif value is not None:
                sanitized[key] = str(value)
        return sanitized

    @staticmethod
    def _query_result_to_documents(result: dict[str, Any]) -> list[Document]:
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        output: list[Document] = []
        for index, text in enumerate(documents):
            metadata = dict(metadatas[index] or {})
            if index < len(distances):
                metadata["distance"] = distances[index]
            output.append(Document(page_content=text, metadata=metadata))
        return output
