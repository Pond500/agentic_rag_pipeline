# agentic_rag_pipeline/mcp_servers/preprocessor_server.py (เวอร์ชัน FINAL)

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any
from fastapi.openapi.utils import get_openapi
from typing import Optional

# --- Import "เครื่องมือ" ของเรา ---
from agentic_rag_pipeline.components import document_preprocessor
from agentic_rag_pipeline.components import metadata_generator
from agentic_rag_pipeline.components import chunker
from agentic_rag_pipeline.components import indexer

# --- สร้าง FastAPI App ---
app = FastAPI(
    title="Agentic RAG Pipeline Tools",
    description="API server providing all pipeline components as callable tools.",
    version="1.0.0"
)

# <--- อัปเกรดฟังก์ชัน Override OpenAPI Schema ---
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    # 1. บังคับเปลี่ยนเวอร์ชัน
    openapi_schema["openapi"] = "3.0.3"

    # 2. (สำคัญ) เพิ่ม servers block เข้าไป
    openapi_schema["servers"] = [
        {
            "url": "http://localhost:8001",  # <<-- URL หลักของ API ของเรา
            "description": "Local Development Server"
        }
    ]

    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi
# <--- สิ้นสุดส่วนที่อัปเกรด ---


# === Tool 1: Document Preprocessor ========================================
class PreprocessRequest(BaseModel):
    file_path: str

class PreprocessResponse(BaseModel):
    clean_text: str
    status: str
    message: str

@app.post("/tools/preprocess_document", response_model=PreprocessResponse, tags=["Pipeline Tools"])
async def preprocess_document_endpoint(request: PreprocessRequest):
    """Tool 1: Takes a file path, returns clean, proofread text."""
    try:
        clean_text = document_preprocessor.process_document(request.file_path)
        if not clean_text:
            return PreprocessResponse(clean_text="", status="error", message="Failed to process document.")
        return PreprocessResponse(clean_text=clean_text, status="success", message="Document processed.")
    except Exception as e:
        return PreprocessResponse(clean_text="", status="error", message=f"Server error: {e}")

# ... (Endpoint ของ Tool 2, 3, 4 ไม่ต้องแก้ไข เหมือนเดิมทุกประการ) ...

# === Tool 2: Metadata Generator ==========================================
class MetadataRequest(BaseModel):
    clean_text: str
    original_filename: str

class MetadataResponse(BaseModel):
    metadata: Dict[str, Any]
    status: str

@app.post("/tools/generate_metadata", response_model=MetadataResponse, tags=["Pipeline Tools"])
async def generate_metadata_endpoint(request: MetadataRequest):
    """Tool 2: Takes clean text, returns structured metadata."""
    metadata = metadata_generator.generate_metadata_for_text(request.clean_text, request.original_filename)
    return MetadataResponse(metadata=metadata, status="success")

# === Tool 3: Chunker =======================================================
class ChunkRequest(BaseModel):
    clean_text: str
    metadata: Dict[str, Any]
    original_filename: str
    strategy: Optional[str] = None # <-- เพิ่ม strategy
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None

class ChunkResponse(BaseModel):
    chunks: List[Dict[str, Any]]
    status: str

@app.post("/tools/create_chunks", response_model=ChunkResponse, tags=["Pipeline Tools"])
async def create_chunks_endpoint(request: ChunkRequest):
    # --- [ใหม่!] ส่งผ่านพารามิเตอร์ใหม่ทั้งหมดเข้าไป ---
    chunks = chunker.create_chunks_for_text(
        text=request.clean_text,
        metadata=request.metadata,
        original_filename=request.original_filename,
        strategy=request.strategy, # <-- ส่ง strategy
        chunk_size=request.chunk_size,
        chunk_overlap=request.chunk_overlap
    )
    return ChunkResponse(chunks=chunks, status="success")

# === Tool 4: Indexer =======================================================
class IndexRequest(BaseModel):
    clean_text: str
    metadata: Dict[str, Any]
    chunks: List[Dict[str, Any]]
    original_filename: str

class IndexResponse(BaseModel):
    success: bool
    message: str

@app.post("/tools/index_document", response_model=IndexResponse, tags=["Pipeline Tools"])
async def index_document_endpoint(request: IndexRequest):
    """Tool 4: Takes all data, creates embeddings, and saves to the database."""
    success = indexer.index_document_and_chunks(
        request.clean_text, request.metadata, request.chunks, request.original_filename
    )
    if success:
        return IndexResponse(success=True, message="Document and chunks indexed successfully.")
    else:
        return IndexResponse(success=False, message="Indexing failed. Check server logs.")

# --- ส่วนสำหรับรัน Server ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)