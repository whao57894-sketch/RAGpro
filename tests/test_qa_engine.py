from langchain_core.documents import Document

from src.embeddings import DeterministicEmbeddingModel
from src.llm import ChatModel
from src.llm import RuleBasedTestChatModel
from src.qa_engine import NO_CONTEXT_ANSWER, QAEngine, build_qa_prompt, extract_sources
from src.vector_store import ChromaVectorStore


class TopContextChatModel(ChatModel):
    def generate(self, prompt: str) -> str:
        if "[CONTEXT]" not in prompt:
            return NO_CONTEXT_ANSWER
        context = prompt.rsplit("[CONTEXT]", 1)[1].split("[QUESTION]", 1)[0]
        for line in context.splitlines():
            line = line.strip()
            if not line or line.startswith("[Source ") or line.startswith("章节：") or line.startswith("字段："):
                continue
            return line
        return NO_CONTEXT_ANSWER


def test_build_prompt_requires_context_only_answer():
    prompt = build_qa_prompt("how to answer?", [])

    assert "Do not use outside knowledge" in prompt
    assert "No relevant context was retrieved." in prompt
    assert "Include source file names" in prompt


def test_qa_engine_returns_answer_and_sources():
    store = ChromaVectorStore(
        embedding_model=DeterministicEmbeddingModel(),
        collection_name="test_qa_engine_returns_answer_and_sources",
    )
    store.add_documents(
        [
            Document(
                page_content="RAG answers must cite source file names.",
                metadata={"file_name": "answer_policy.txt", "chunk_index": 0},
            ),
            Document(
                page_content="Vector databases store embeddings and return Top-K similar chunks.",
                metadata={"file_name": "vector_policy.txt", "chunk_index": 1},
            ),
        ]
    )

    engine = QAEngine(vector_store=store, chat_model=RuleBasedTestChatModel(), top_k=2)
    result = engine.answer("Why must answers cite source file names?")

    assert "cite source file names" in result.answer.lower()
    assert result.sources
    assert any(source.file_name == "answer_policy.txt" for source in result.sources)
    assert "answer_policy.txt" in result.prompt


def test_qa_engine_no_context_returns_fallback():
    class EmptyStore:
        def similarity_search(self, query: str, top_k: int = 4):
            return []

    engine = QAEngine(vector_store=EmptyStore(), chat_model=RuleBasedTestChatModel())
    result = engine.answer("missing question")

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


def test_qa_engine_handles_colloquial_field_question_with_keyword_provider():
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
        collection_name="test_qa_engine_colloquial_field",
    )
    store.add_documents(documents)

    engine = QAEngine(
        vector_store=store,
        chat_model=TopContextChatModel(),
        top_k=2,
        retrieval_top_k=6,
        keyword_documents_provider=lambda: documents,
    )
    result = engine.answer("我的姓名是什么")

    assert "姓名：张三" in result.answer
    assert result.sources
    assert result.sources[0].file_name == "profile.docx"
    assert len(result.sources) == 1
