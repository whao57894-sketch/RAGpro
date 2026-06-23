from pathlib import Path


def test_streamlit_app_contains_required_ui_elements():
    source = Path("frontend/streamlit_app.py").read_text(encoding="utf-8")

    assert "file_uploader" in source
    assert "st.chat_input" in source
    assert "st.chat_message" in source
    assert "requests.post" in source
    assert "/documents/upload" in source
    assert "/qa/ask" in source
    assert "/documents/clear" in source
