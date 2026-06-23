from pathlib import Path


def main() -> None:
    app_path = Path("frontend/streamlit_app.py")
    if not app_path.exists():
        raise SystemExit("Missing frontend/streamlit_app.py")

    source = app_path.read_text(encoding="utf-8")
    required_fragments = [
        "file_uploader",
        "st.chat_input",
        "st.chat_message",
        "requests.post",
        "/documents/upload",
        "/qa/ask",
        "/documents/clear",
        "render_sources",
    ]
    missing = [fragment for fragment in required_fragments if fragment not in source]
    if missing:
        raise SystemExit(f"Missing Streamlit fragments: {missing}")

    print("Streamlit app source OK")


if __name__ == "__main__":
    main()
