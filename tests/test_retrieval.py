from langchain_core.documents import Document

from src.embeddings import DeterministicEmbeddingModel
from src.retrieval import BM25Retriever, HybridRetriever, evaluate_retrieval
from src.vector_store import ChromaVectorStore


def test_bm25_search_returns_keyword_match():
    documents = [
        Document(
            page_content="向量数据库负责相似度检索和 Top-K 召回。",
            metadata={"file_name": "vector.txt", "chunk_index": 0},
        ),
        Document(
            page_content="回答需要标注来源文件名和出处。",
            metadata={"file_name": "answer.txt", "chunk_index": 1},
        ),
    ]

    retriever = BM25Retriever(documents)
    results = retriever.search("相似度检索", top_k=1)

    assert results[0].document.metadata["file_name"] == "vector.txt"


def test_hybrid_retriever_merges_and_deduplicates():
    documents = [
        Document(
            page_content="向量数据库负责相似度检索和 Top-K 召回。",
            metadata={"file_name": "vector.txt", "chunk_index": 0},
        ),
        Document(
            page_content="回答需要标注来源文件名和出处。",
            metadata={"file_name": "answer.txt", "chunk_index": 1},
        ),
    ]
    store = ChromaVectorStore(
        embedding_model=DeterministicEmbeddingModel(),
        collection_name="test_hybrid_retriever_merges_and_deduplicates",
    )
    store.add_documents(documents)

    retriever = HybridRetriever(store, documents)
    results = retriever.search("相似度检索", top_k=2)

    file_names = [document.metadata["file_name"] for document in results]
    assert file_names[0] == "vector.txt"
    assert len(file_names) == len(set(file_names))


def test_evaluate_retrieval_returns_accuracy():
    documents = [
        Document(
            page_content="向量数据库负责相似度检索和 Top-K 召回。",
            metadata={"file_name": "vector.txt", "chunk_index": 0},
        ),
        Document(
            page_content="回答需要标注来源文件名和出处。",
            metadata={"file_name": "answer.txt", "chunk_index": 1},
        ),
    ]
    retriever = BM25Retriever(documents)
    metrics = evaluate_retrieval(
        retriever,
        [{"question": "相似度检索", "expected_file": "vector.txt"}],
    )

    assert metrics["total"] == 1
    assert metrics["hits"] == 1
    assert metrics["accuracy"] == 1.0
