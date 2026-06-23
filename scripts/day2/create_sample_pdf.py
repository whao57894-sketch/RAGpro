from pathlib import Path


PDF_TEXT_LINES = [
    "Enterprise RAG Test Document",
    "Policy: Employees can ask questions based on uploaded documents.",
    "Answer requirements: cite document source and avoid unsupported claims.",
    "Contact: knowledge-base team.",
]


def _pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def build_pdf_bytes(lines: list[str]) -> bytes:
    text_commands = ["BT", "/F1 12 Tf", "72 760 Td", "14 TL"]
    for index, line in enumerate(lines):
        if index:
            text_commands.append("T*")
        text_commands.append(f"({_pdf_escape(line)}) Tj")
    text_commands.append("ET")
    stream = "\n".join(text_commands).encode("ascii")

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream",
    ]

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for number, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{number} 0 obj\n".encode("ascii"))
        pdf.extend(obj)
        pdf.extend(b"\nendobj\n")

    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    pdf.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n".encode("ascii")
    )
    return bytes(pdf)


def main() -> None:
    output_path = Path("data/samples/enterprise_policy_test.pdf")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(build_pdf_bytes(PDF_TEXT_LINES))
    print(f"Created {output_path}")


if __name__ == "__main__":
    main()
