from pathlib import Path

from pypdf import PdfReader


EXPECTED_PHRASES = [
    "Enterprise RAG Test Document",
    "cite document source",
    "avoid unsupported claims",
]


def main() -> None:
    pdf_path = Path("data/samples/enterprise_policy_test.pdf")
    if not pdf_path.exists():
        raise SystemExit(f"Missing sample PDF: {pdf_path}")

    reader = PdfReader(str(pdf_path))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    missing = [phrase for phrase in EXPECTED_PHRASES if phrase not in text]
    if missing:
        raise SystemExit(f"PDF text extraction failed, missing: {missing}\nExtracted text:\n{text}")

    output_path = Path("docs/day2/pdf_parse_result.txt")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    print("PDF parsing OK")
    print(text)


if __name__ == "__main__":
    main()
