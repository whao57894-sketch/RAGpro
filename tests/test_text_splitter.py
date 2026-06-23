import pytest
from langchain_core.documents import Document

from src.text_splitter import (
    CHINESE_SEPARATORS,
    SplitConfig,
    compare_split_configs,
    split_documents,
)


def test_split_documents_keeps_metadata_and_content():
    document = Document(
        page_content=(
            "第一段说明企业知识库的目标。系统需要基于文档回答问题，并且标注出处。\n\n"
            "第二段说明文本切分的重要性。切分太短会丢失上下文，切分太长会影响检索精度。"
        ),
        metadata={"file_name": "policy.txt", "file_path": "data/policy.txt"},
    )

    chunks = split_documents([document], chunk_size=45, chunk_overlap=10)

    assert len(chunks) > 1
    assert all(isinstance(chunk, Document) for chunk in chunks)
    assert "企业知识库" in "".join(chunk.page_content for chunk in chunks)
    for index, chunk in enumerate(chunks):
        assert chunk.metadata["file_name"] == "policy.txt"
        assert chunk.metadata["chunk_index"] == index
        assert chunk.metadata["chunk_size"] == 45
        assert chunk.metadata["chunk_overlap"] == 10
        assert chunk.metadata["chunk_length"] == len(chunk.page_content)


def test_compare_split_configs_returns_metrics():
    document = Document(
        page_content="这是一个用于测试切分参数的中文段落。" * 20,
        metadata={},
    )
    results = compare_split_configs(
        [document],
        [
            SplitConfig(chunk_size=80, chunk_overlap=10),
            SplitConfig(chunk_size=120, chunk_overlap=20),
        ],
    )

    assert len(results) == 2
    assert results[0]["chunk_count"] >= results[1]["chunk_count"]
    assert all("quality_score" in result for result in results)


def test_invalid_split_params():
    with pytest.raises(ValueError):
        split_documents([], chunk_size=100, chunk_overlap=100)


def test_chinese_separators_include_sentence_punctuation():
    for separator in ["\n\n", "。", "！", "？", "，"]:
        assert separator in CHINESE_SEPARATORS
