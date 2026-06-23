from pathlib import Path

import pytest
from langchain_core.documents import Document

from scripts.day3.create_sample_documents import main as create_sample_documents
from src.document_parser import UnsupportedDocumentTypeError, parse_document, parse_documents


EXPECTED_PHRASES = [
    "Enterprise document parsing sample.",
    "Employees can search policies",
    "Parsed documents must keep file metadata",
]


def test_parse_pdf_docx_txt_documents():
    create_sample_documents()
    sample_dir = Path("data/day3_samples")
    paths = [
        sample_dir / "sample_policy.pdf",
        sample_dir / "sample_manual.docx",
        sample_dir / "sample_faq.txt",
    ]

    documents = parse_documents(paths)

    assert documents
    assert all(isinstance(document, Document) for document in documents)

    parsed_text = "\n".join(document.page_content for document in documents)
    for phrase in EXPECTED_PHRASES:
        assert phrase in parsed_text

    file_names = {document.metadata["file_name"] for document in documents}
    assert file_names == {"sample_policy.pdf", "sample_manual.docx", "sample_faq.txt"}

    for document in documents:
        assert Path(document.metadata["file_path"]).exists()
        assert document.metadata["file_extension"] in {".pdf", ".docx", ".txt"}
        assert isinstance(document.metadata["document_index"], int)


def test_parse_unsupported_file_type(tmp_path):
    file_path = tmp_path / "sample.xlsx"
    file_path.write_text("unsupported", encoding="utf-8")

    with pytest.raises(UnsupportedDocumentTypeError):
        parse_document(file_path)
