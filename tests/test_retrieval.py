from langchain_core.documents import Document

from src.embeddings import DeterministicEmbeddingModel
from src.retrieval import BM25Retriever, HybridRetriever, evaluate_retrieval, expand_query, tokenize_for_retrieval
from src.vector_store import ChromaVectorStore


def test_tokenize_for_retrieval_supports_short_chinese_queries():
    tokens = tokenize_for_retrieval("项目名称")

    assert "项" in tokens
    assert "项目" in tokens
    assert "名称" in tokens
    assert "项目名称" in tokens


def test_expand_query_handles_colloquial_field_question():
    queries = expand_query("我的姓名是什么")

    assert "姓名" in queries
    assert "姓名：" in queries


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


def test_bm25_search_handles_template_field_names():
    documents = [
        Document(
            page_content="项目名称：Alpha 管理系统。功能模块：录入、统计、导出。",
            metadata={"file_name": "project.txt", "chunk_index": 0},
        ),
        Document(
            page_content="普通说明：本文档介绍使用流程。",
            metadata={"file_name": "manual.txt", "chunk_index": 1},
        ),
    ]

    retriever = BM25Retriever(documents)
    results = retriever.search("项目名称", top_k=1)

    assert results
    assert results[0].document.metadata["file_name"] == "project.txt"


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


def test_hybrid_retriever_handles_colloquial_field_query():
    documents = [
        Document(
            page_content="章节：项目成员\n字段：姓名\n姓名：张三",
            metadata={"file_name": "profile.docx", "chunk_index": 0, "section_type": "table_row", "field_keys": "姓名"},
        ),
        Document(
            page_content="章节：项目成员\n字段：模块\n模块：检索优化",
            metadata={"file_name": "profile.docx", "chunk_index": 1, "section_type": "table_row", "field_keys": "模块"},
        ),
    ]
    store = ChromaVectorStore(
        embedding_model=DeterministicEmbeddingModel(),
        collection_name="test_hybrid_retriever_handles_colloquial_field_query",
    )
    store.add_documents(documents)

    retriever = HybridRetriever(store, documents)
    results = retriever.search("我的姓名是什么", top_k=2)

    assert results
    assert "姓名：张三" in results[0].page_content


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
