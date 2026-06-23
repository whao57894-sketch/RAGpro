import os

import requests
import streamlit as st


DEFAULT_API_BASE_URL = os.getenv("RAG_API_BASE_URL", "http://127.0.0.1:8000")


def init_state() -> None:
    if "api_base_url" not in st.session_state:
        st.session_state.api_base_url = DEFAULT_API_BASE_URL
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "documents" not in st.session_state:
        st.session_state.documents = []


def api_url(path: str) -> str:
    return st.session_state.api_base_url.rstrip("/") + path


def get_documents() -> list[dict]:
    response = requests.get(api_url("/documents"), timeout=20)
    response.raise_for_status()
    return response.json()["documents"]


def upload_file(uploaded_file) -> dict:
    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type or "application/octet-stream")}
    response = requests.post(api_url("/documents/upload"), files=files, timeout=120)
    response.raise_for_status()
    return response.json()


def ask_question(question: str) -> dict:
    response = requests.post(api_url("/qa/ask"), json={"question": question}, timeout=120)
    response.raise_for_status()
    return response.json()


def clear_knowledge_base() -> dict:
    response = requests.delete(api_url("/documents/clear"), timeout=30)
    response.raise_for_status()
    return response.json()


def refresh_documents() -> None:
    try:
        st.session_state.documents = get_documents()
    except requests.RequestException:
        st.session_state.documents = []


def render_style() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.5rem;
            max-width: 1180px;
        }
        [data-testid="stSidebar"] {
            background: #f6f8fb;
            border-right: 1px solid #e5e7eb;
        }
        .metric-card {
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 12px 14px;
            background: #ffffff;
        }
        .source-pill {
            display: inline-block;
            margin: 4px 6px 0 0;
            padding: 4px 8px;
            border-radius: 999px;
            background: #eef2ff;
            color: #3730a3;
            font-size: 12px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> None:
    st.sidebar.header("Knowledge Base")
    st.sidebar.text_input("Backend API", key="api_base_url")

    uploaded_files = st.sidebar.file_uploader(
        "Upload PDF, DOCX or TXT",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
    )

    if uploaded_files and st.sidebar.button("Upload and index", type="primary", use_container_width=True):
        for uploaded_file in uploaded_files:
            try:
                with st.spinner(f"Uploading {uploaded_file.name}"):
                    upload_file(uploaded_file)
            except requests.HTTPError as exc:
                detail = exc.response.text if exc.response is not None else str(exc)
                st.sidebar.error(f"{uploaded_file.name}: {detail}")
            except requests.RequestException as exc:
                st.sidebar.error(f"Backend unavailable: {exc}")
        refresh_documents()

    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("Refresh", use_container_width=True):
            refresh_documents()
    with col2:
        if st.button("Clear", use_container_width=True):
            try:
                clear_knowledge_base()
                st.session_state.messages = []
                refresh_documents()
                st.sidebar.success("Cleared")
            except requests.RequestException as exc:
                st.sidebar.error(f"Clear failed: {exc}")

    st.sidebar.subheader("Indexed Documents")
    if not st.session_state.documents:
        st.sidebar.caption("No documents indexed")
    for document in st.session_state.documents:
        st.sidebar.markdown(
            f"**{document['file_name']}**  \n"
            f"{document['chunk_count']} chunks · {document['vector_count']} vectors"
        )


def render_header() -> None:
    st.title("Enterprise Document QA")
    st.caption("Upload internal documents, ask questions, and inspect cited source files.")

    total_docs = len(st.session_state.documents)
    total_chunks = sum(document.get("chunk_count", 0) for document in st.session_state.documents)
    col1, col2, col3 = st.columns(3)
    col1.markdown(f"<div class='metric-card'><b>{total_docs}</b><br/>Documents</div>", unsafe_allow_html=True)
    col2.markdown(f"<div class='metric-card'><b>{total_chunks}</b><br/>Chunks</div>", unsafe_allow_html=True)
    col3.markdown(
        f"<div class='metric-card'><b>{st.session_state.api_base_url}</b><br/>Backend</div>",
        unsafe_allow_html=True,
    )


def render_chat() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                render_sources(message["sources"])

    question = st.chat_input("Ask a question about uploaded documents")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            response = ask_question(question)
            answer = response["answer"]
            sources = response.get("sources", [])
            st.markdown(answer)
            render_sources(sources)
        except requests.HTTPError as exc:
            answer = exc.response.json().get("detail", exc.response.text) if exc.response is not None else str(exc)
            sources = []
            st.warning(answer)
        except requests.RequestException as exc:
            answer = f"Backend unavailable: {exc}"
            sources = []
            st.error(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    pills = []
    for source in sources:
        label = source.get("file_name", "unknown")
        chunk_index = source.get("chunk_index")
        if chunk_index is not None:
            label = f"{label} · chunk {chunk_index}"
        pills.append(f"<span class='source-pill'>{label}</span>")
    st.markdown("".join(pills), unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(page_title="Enterprise Document QA", page_icon="DOC", layout="wide")
    init_state()
    render_style()
    refresh_documents()
    render_sidebar()
    render_header()
    render_chat()


if __name__ == "__main__":
    main()
