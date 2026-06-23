from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.document_parser import UnsupportedDocumentTypeError, parse_document
from src.embeddings import DeterministicEmbeddingModel
from src.llm import ChatModel, RuleBasedTestChatModel, ZhipuChatModel
from src.qa_engine import QAEngine
from src.text_splitter import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, split_documents
from src.vector_store import ChromaVectorStore


UPLOAD_DIR = Path("data/uploads")


class UploadResponse(BaseModel):
    document_id: str
    file_name: str
    saved_path: str
    parsed_documents: int
    chunk_count: int
    vector_count: int


class DocumentInfo(BaseModel):
    document_id: str
    file_name: str
    saved_path: str
    parsed_documents: int
    chunk_count: int
    vector_count: int


class DocumentListResponse(BaseModel):
    total: int
    documents: list[DocumentInfo]


class AskRequest(BaseModel):
    question: str


class SourceResponse(BaseModel):
    file_name: str
    chunk_index: int | None = None
    distance: float | None = None


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceResponse]


class ClearResponse(BaseModel):
    cleared_documents: int
    cleared_vectors: int


def create_app(
    *,
    embedding_model=None,
    chat_model: ChatModel | None = None,
    collection_name: str = "api_uploaded_documents",
) -> FastAPI:
    app = FastAPI(title="Enterprise RAG API", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    resolved_embedding_model = embedding_model or DeterministicEmbeddingModel()
    resolved_chat_model = chat_model or _build_default_chat_model()
    app.state.vector_store = ChromaVectorStore(
        embedding_model=resolved_embedding_model,
        collection_name=collection_name,
    )
    app.state.documents = {}
    app.state.document_vectors = {}
    app.state.qa_engine = QAEngine(
        vector_store=app.state.vector_store,
        chat_model=resolved_chat_model,
    )

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/documents/upload", response_model=UploadResponse)
    async def upload_document(file: UploadFile = File(...)):
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in {".pdf", ".docx", ".txt"}:
            raise HTTPException(status_code=400, detail="Only PDF, DOCX and TXT files are supported")

        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        document_id = uuid4().hex
        safe_file_name = Path(file.filename or f"upload{suffix}").name
        saved_path = UPLOAD_DIR / f"{document_id}_{safe_file_name}"

        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        saved_path.write_bytes(content)

        try:
            parsed_documents = parse_document(saved_path)
            chunks = split_documents(
                parsed_documents,
                chunk_size=DEFAULT_CHUNK_SIZE,
                chunk_overlap=DEFAULT_CHUNK_OVERLAP,
            )
            for chunk in chunks:
                chunk.metadata["document_id"] = document_id
                chunk.metadata["file_name"] = safe_file_name
                chunk.metadata["original_file_name"] = safe_file_name
            vector_ids = app.state.vector_store.add_documents(chunks)
            app.state.qa_engine.clear_cache()
        except UnsupportedDocumentTypeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to ingest document: {exc}") from exc

        info = DocumentInfo(
            document_id=document_id,
            file_name=safe_file_name,
            saved_path=str(saved_path),
            parsed_documents=len(parsed_documents),
            chunk_count=len(chunks),
            vector_count=len(vector_ids),
        )
        app.state.documents[document_id] = info
        app.state.document_vectors[document_id] = vector_ids
        return UploadResponse(**info.model_dump())

    @app.get("/documents", response_model=DocumentListResponse)
    def list_documents():
        documents = list(app.state.documents.values())
        return DocumentListResponse(total=len(documents), documents=documents)

    @app.delete("/documents/clear", response_model=ClearResponse)
    def clear_documents():
        cleared_documents = len(app.state.documents)
        vector_ids: list[str] = []
        for ids in app.state.document_vectors.values():
            vector_ids.extend(ids)

        if vector_ids:
            app.state.vector_store.collection.delete(ids=vector_ids)

        app.state.documents.clear()
        app.state.document_vectors.clear()
        app.state.qa_engine.clear_cache()
        return ClearResponse(
            cleared_documents=cleared_documents,
            cleared_vectors=len(vector_ids),
        )

    @app.post("/qa/ask", response_model=AskResponse)
    def ask_question(request: AskRequest):
        if not app.state.documents:
            raise HTTPException(status_code=400, detail="Knowledge base is empty. Upload documents first.")

        try:
            result = app.state.qa_engine.answer(request.question)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"QA failed: {exc}") from exc

        return AskResponse(
            question=result.question,
            answer=result.answer,
            sources=[
                SourceResponse(
                    file_name=source.file_name,
                    chunk_index=source.chunk_index,
                    distance=source.distance,
                )
                for source in result.sources
            ],
        )

    return app


def _build_default_chat_model() -> ChatModel:
    try:
        return ZhipuChatModel()
    except Exception:
        return RuleBasedTestChatModel()


app = create_app()
