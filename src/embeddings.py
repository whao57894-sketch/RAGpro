import hashlib
import os
import re
from abc import ABC, abstractmethod
from typing import Sequence

from dotenv import load_dotenv
from zhipuai import ZhipuAI


class EmbeddingModel(ABC):
    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed document chunks."""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a user query."""


class ZhipuEmbeddingModel(EmbeddingModel):
    def __init__(self, api_key: str | None = None, model: str = "embedding-2") -> None:
        load_dotenv()
        self.model = model
        resolved_api_key = api_key or os.getenv("ZHIPUAI_API_KEY")
        if not resolved_api_key or resolved_api_key == "your_zhipu_api_key_here":
            raise ValueError("ZHIPUAI_API_KEY is missing")
        self.client = ZhipuAI(api_key=resolved_api_key)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_text(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_text(text)

    def _embed_text(self, text: str) -> list[float]:
        response = self.client.embeddings.create(model=self.model, input=text)
        return list(response.data[0].embedding)


class DeterministicEmbeddingModel(EmbeddingModel):
    """Small deterministic embedding model for local tests."""

    def __init__(self, dimensions: int = 16) -> None:
        self.dimensions = dimensions

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in _tokenize(text):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = digest[0] % self.dimensions
            weight = 1.0 + digest[1] / 255.0
            vector[index] += weight
        norm = sum(value * value for value in vector) ** 0.5
        if norm == 0:
            return vector
        return [round(value / norm, 8) for value in vector]


def _tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]", text.lower())
    bigrams = [f"{tokens[index]}{tokens[index + 1]}" for index in range(len(tokens) - 1)]
    return tokens + bigrams
