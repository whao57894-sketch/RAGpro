import subprocess
import sys


CHECKS = [
    ("Create sample PDF", ["scripts/day2/create_sample_pdf.py"]),
    ("PDF parsing", ["scripts/day2/verify_pdf_parsing.py"]),
    ("Zhipu API", ["scripts/day2/verify_zhipu_api.py"]),
    ("ChromaDB", ["scripts/day2/verify_chroma.py"]),
]


def main() -> None:
    failures: list[tuple[str, int]] = []
    for name, command in CHECKS:
        print(f"\n== {name} ==")
        result = subprocess.run([sys.executable, *command], check=False)
        if result.returncode != 0:
            failures.append((name, result.returncode))

    if failures:
        print("\nFailed checks:")
        for name, code in failures:
            print(f"- {name}: exit code {code}")
        raise SystemExit(1)

    print("\nAll day 2 checks passed")


if __name__ == "__main__":
    main()
