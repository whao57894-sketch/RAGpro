from langchain_core.documents import Document

from src.embeddings import DeterministicEmbeddingModel
from src.llm import RuleBasedTestChatModel
from src.qa_engine import NO_CONTEXT_ANSWER, QAEngine, build_qa_prompt, extract_sources
from src.vector_store import ChromaVectorStore


def test_build_prompt_requires_context_only_answer():
    prompt = build_qa_prompt("如何回答？", [])

    assert "只能使用检索资料中的信息回答" in prompt
    assert "没有在已上传文档中找到相关信息" in prompt
    assert "引用来源文件名" in prompt


def test_qa_engine_returns_answer_and_sources():
    store = ChromaVectorStore(
        embedding_model=DeterministicEmbeddingModel(),
        collection_name="test_qa_engine_returns_answer_and_sources",
    )
    store.add_documents(
        [
            Document(
                page_content="所有问答结果必须基于检索到的文本块，并标注文件名和出处。",
                metadata={"file_name": "answer_policy.txt", "chunk_index": 0},
            ),
            Document(
                page_content="向量数据库负责存储文本块 embedding，并按相似度返回 Top-K 结果。",
                metadata={"file_name": "vector_policy.txt", "chunk_index": 1},
            ),
        ]
    )

    engine = QAEngine(vector_store=store, chat_model=RuleBasedTestChatModel(), top_k=2)
    result = engine.answer("回答为什么要标注出处？")

    assert "标注文件名和出处" in result.answer
    assert result.sources
    assert any(source.file_name == "answer_policy.txt" for source in result.sources)
    assert "answer_policy.txt" in result.prompt


def test_qa_engine_no_context_returns_fallback():
    class EmptyStore:
        def similarity_search(self, query: str, top_k: int = 4):
            return []

    engine = QAEngine(vector_store=EmptyStore(), chat_model=RuleBasedTestChatModel())
    result = engine.answer("不存在的问题")

    assert result.answer == NO_CONTEXT_ANSWER
    assert result.sources == []


def test_extract_sources_deduplicates_by_file_and_chunk():
    documents = [
        Document(page_content="a", metadata={"file_name": "a.txt", "chunk_index": 1, "distance": 0.1}),
        Document(page_content="b", metadata={"file_name": "a.txt", "chunk_index": 1, "distance": 0.2}),
    ]

    sources = extract_sources(documents)

    assert len(sources) == 1
    assert sources[0].file_name == "a.txt"
    assert sources[0].chunk_index == 1
