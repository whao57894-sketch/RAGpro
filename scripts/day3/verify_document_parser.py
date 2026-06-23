from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.documents import Document

from src.document_parser import parse_document, parse_documents


EXPECTED_PHRASES = [
    "Enterprise document parsing sample.",
    "Employees can search policies",
    "Parsed documents must keep file metadata",
]


def assert_parsed(path: Path) -> None:
    documents = parse_document(path)
    if not documents:
        raise SystemExit(f"No documents parsed from {path}")
    if not all(isinstance(document, Document) for document in documents):
        raise SystemExit(f"Parser did not return Document objects for {path}")

    text = "\n".join(document.page_content for document in documents)
    missing = [phrase for phrase in EXPECTED_PHRASES if phrase not in text]
    if missing:
        raise SystemExit(f"Missing expected text from {path}: {missing}\n{text}")

    for document in documents:
        metadata = document.metadata
        for key in ["file_name", "file_path", "file_extension", "document_index"]:
            if key not in metadata:
                raise SystemExit(f"Missing metadata {key} in {path}: {metadata}")

    print(f"OK: {path} -> {len(documents)} Document object(s)")


def main() -> None:
    sample_dir = Path("data/day3_samples")
    paths = [
        sample_dir / "sample_policy.pdf",
        sample_dir / "sample_manual.docx",
        sample_dir / "sample_faq.txt",
    ]

    for path in paths:
        assert_parsed(path)

    combined = parse_documents(paths)
    print(f"Combined parse OK: {len(combined)} Document object(s)")


if __name__ == "__main__":
    main()
