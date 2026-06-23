import pytest
from langchain_core.documents import Document

from src.embeddings import DeterministicEmbeddingModel
from src.vector_store import ChromaVectorStore


def test_chroma_vector_store_add_and_search():
    store = ChromaVectorStore(
        embedding_model=DeterministicEmbeddingModel(),
        collection_name="test_vector_store_add_and_search",
    )
    documents = [
        Document(
            page_content="PDF 文档解析后会进入文本切分流程。",
            metadata={"file_name": "parser.txt", "chunk_index": 0},
        ),
        Document(
            page_content="Chroma 向量数据库用于相似度检索和 Top-K 召回。",
            metadata={"file_name": "vector.txt", "chunk_index": 1},
        ),
        Document(
            page_content="智谱 embedding-2 可以把文本转换成向量。",
            metadata={"file_name": "embedding.txt", "chunk_index": 2},
        ),
    ]

    inserted_ids = store.add_documents(documents)
    results = store.similarity_search("Chroma Top-K 相似度检索", top_k=2)

    assert len(inserted_ids) == 3
    assert store.count() == 3
    assert len(results) == 2
    assert "Chroma" in results[0].page_content
    assert results[0].metadata["file_name"] == "vector.txt"
    assert "distance" in results[0].metadata


def test_empty_add_returns_empty_list():
    store = ChromaVectorStore(
        embedding_model=DeterministicEmbeddingModel(),
        collection_name="test_vector_store_empty_add",
    )

    assert store.add_documents([]) == []


def test_invalid_top_k_raises_error():
    store = ChromaVectorStore(
        embedding_model=DeterministicEmbeddingModel(),
        collection_name="test_vector_store_invalid_top_k",
    )

    with pytest.raises(ValueError, match="top_k"):
        store.similarity_search("query", top_k=0)
