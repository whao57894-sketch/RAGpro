import json
import ast
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT_DIR / "docs" / "day15"
REPORT_PATH = REPORT_DIR / "code_quality_report.md"
RESULTS_PATH = REPORT_DIR / "code_quality_results.json"

SOURCE_DIRS = ["src", "api", "frontend", "scripts", "tests"]
SECRET_PATTERNS = [
    re.compile(r"zhipuai_api_key\s*=\s*['\"][^'\"]+['\"]", re.IGNORECASE),
    re.compile(r"api_key\s*=\s*['\"](?!your_)[^'\"]{12,}['\"]", re.IGNORECASE),
    re.compile(r"sk-[a-zA-Z0-9_-]{16,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
]
ALLOWLIST_FILES = {".env.example"}


def run_quality_check(run_tests: bool = True) -> dict[str, Any]:
    python_files = _python_files()
    compile_errors = _compile_python_files(python_files)
    secret_findings = _scan_for_secrets()
    test_result = _run_core_tests() if run_tests else {"returncode": None, "stdout": "", "stderr": ""}

    results = {
        "python_file_count": len(python_files),
        "compile_errors": compile_errors,
        "secret_findings": secret_findings,
        "tests": test_result,
        "passed": not compile_errors and not secret_findings and test_result.get("returncode") in {0, None},
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    REPORT_PATH.write_text(_render_report(results), encoding="utf-8")
    return results


def _python_files() -> list[Path]:
    files: list[Path] = []
    for directory in SOURCE_DIRS:
        root = ROOT_DIR / directory
        if root.exists():
            files.extend(
                path
                for path in root.rglob("*.py")
                if "__pycache__" not in path.parts and ".venv" not in path.parts
            )
    return sorted(files)


def _compile_python_files(files: list[Path]) -> list[dict[str, str]]:
    errors = []
    for path in files:
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:
            errors.append({"file": str(path.relative_to(ROOT_DIR)), "error": str(exc)})
    return errors


def _scan_for_secrets() -> list[dict[str, str]]:
    findings = []
    for path in _text_files():
        relative = path.relative_to(ROOT_DIR)
        if relative.name in ALLOWLIST_FILES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for pattern in SECRET_PATTERNS:
            for match in pattern.finditer(text):
                findings.append(
                    {
                        "file": str(relative),
                        "pattern": pattern.pattern,
                        "match": _redact(match.group(0)),
                    }
                )
    return findings


def _text_files() -> list[Path]:
    suffixes = {".py", ".md", ".txt", ".json", ".toml", ".yaml", ".yml", ".env", ".example"}
    ignored_parts = {".venv", ".venv311", "vendor_wheels", "local_packages", "__pycache__"}
    files = []
    for path in ROOT_DIR.rglob("*"):
        if not path.is_file():
            continue
        if any(part in ignored_parts for part in path.parts):
            continue
        if path.suffix.lower() in suffixes or path.name.startswith(".env"):
            files.append(path)
    return files


def _run_core_tests() -> dict[str, Any]:
    command = [
        str(ROOT_DIR / ".venv311" / "Scripts" / "python.exe"),
        "-m",
        "pytest",
        "tests/test_document_parser.py",
        "tests/test_text_splitter.py",
        "tests/test_retrieval.py",
        "tests/test_vector_store.py",
        "-q",
    ]
    completed = subprocess.run(command, cwd=ROOT_DIR, text=True, capture_output=True, timeout=120)
    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _redact(value: str) -> str:
    if len(value) <= 12:
        return "***"
    return f"{value[:6]}...{value[-4:]}"


def _render_report(results: dict[str, Any]) -> str:
    lines = [
        "# Day 15 Code Quality And Unit Test Report",
        "",
        "## Summary",
        "",
        f"- Python files checked: {results['python_file_count']}",
        f"- Compile errors: {len(results['compile_errors'])}",
        f"- Secret findings: {len(results['secret_findings'])}",
        f"- Core test return code: {results['tests'].get('returncode')}",
        f"- Overall status: {'PASS' if results['passed'] else 'REVIEW'}",
        "",
        "## Core Test Output",
        "",
        "```text",
        results["tests"].get("stdout", "").strip(),
        "```",
        "",
        "## Sensitive Information Check",
        "",
    ]
    if results["secret_findings"]:
        for finding in results["secret_findings"]:
            lines.append(f"- {finding['file']}: {finding['match']}")
    else:
        lines.append("- No hard-coded secrets found in scanned project files.")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Real API keys should stay in `.env`, which is not printed by this report.",
            "- `.env.example` contains placeholders only and is allowed.",
            "- Formatting was kept within the existing toolchain because `black` is not installed in the local environment.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    results = run_quality_check()
    print(f"Python files checked: {results['python_file_count']}")
    print(f"Compile errors: {len(results['compile_errors'])}")
    print(f"Secret findings: {len(results['secret_findings'])}")
    print(f"Core test return code: {results['tests'].get('returncode')}")
    print(f"Report: {REPORT_PATH.relative_to(ROOT_DIR)}")
    print(f"Raw results: {RESULTS_PATH.relative_to(ROOT_DIR)}")


if __name__ == "__main__":
    main()
