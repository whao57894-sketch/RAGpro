from importlib import import_module
from pathlib import Path


REQUIRED_MODULES = [
    "dotenv",
    "fastapi",
    "langchain",
    "langchain_community",
    "langchain_core",
    "pydantic",
    "zhipuai",
    "pypdf",
    "docx",
]

REQUIRED_DIRS = ["src", "api", "frontend", "data", "tests"]


def main() -> None:
    project_root = Path(__file__).resolve().parent

    missing_dirs = [name for name in REQUIRED_DIRS if not (project_root / name).is_dir()]
    if missing_dirs:
        raise SystemExit(f"Missing directories: {', '.join(missing_dirs)}")

    for module_name in REQUIRED_MODULES:
        import_module(module_name)

    env_file = project_root / ".env"
    if not env_file.exists():
        raise SystemExit("Missing .env file")

    print("Environment verification OK")


if __name__ == "__main__":
    main()
