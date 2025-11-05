# agentic_rag_pipeline/mcp_servers/preprocessor_server.py

import uvicorn
from fastapi import FastAPI, BackgroundTasks, UploadFile, File, Form # <-- [ใหม่!] ขั้นตอนที่ 4: เพิ่ม Import
from pydantic import BaseModel
from typing import List, Dict, Any
from fastapi.openapi.utils import get_openapi
from typing import Optional
import tempfile # <-- [ใหม่!] ขั้นตอนที่ 4: เพิ่ม Import
import os       # <-- [ใหม่!] ขั้นตอนที่ 4: เพิ่ม Import

# --- Import "เครื่องมือ" ของเรา ---
from agentic_rag_pipeline.components import document_preprocessor
from agentic_rag_pipeline.components import metadata_generator
from agentic_rag_pipeline.components import chunker
from agentic_rag_pipeline.components import indexer

# --- [ใหม่!] ขั้นตอนที่ 4: Import "โรงงาน" (Graph) ---
from agentic_rag_pipeline.graph_agent.graph import graph_app

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
    
    # --- [V2+V5] ---
    layout_map: Dict[str, Any]
    retry_instructions: Dict[str, Any]

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
        layout_map=request.layout_map, # <-- [V2]
        retry_instructions=request.retry_instructions # <-- [V5]
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

# ==============================================================================
# [ใหม่!] ขั้นตอนที่ 4: สร้าง Dify Integration Endpoint
# ==============================================================================
class DifyProcessResponse(BaseModel):
    status: str
    message: str
    filename: str

def run_graph_in_background(initial_state: Dict[str, Any]):
    """Helper function to run the graph invocation."""
    print(f"--- Background task started for: {initial_state.get('original_filename')} ---")
    try:
        final_state = graph_app.invoke(initial_state)
        if final_state.get("error_message"):
            print(f"--- ❌ Background task FAILED for: {initial_state.get('original_filename')} ---")
            print(f"    Error: {final_state['error_message']}")
        else:
            print(f"--- ✅ Background task COMPLETED for: {initial_state.get('original_filename')} ---")
    except Exception as e:
        print(f"--- ❌ CRITICAL BACKGROUND ERROR for: {initial_state.get('original_filename')}: {e} ---")
    finally:
        # ลบไฟล์ temp ทิ้งหลังจากประมวลผลเสร็จ
        if os.path.exists(initial_state["file_path"]):
            os.remove(initial_state["file_path"])
            print(f"   -> Removed temp file: {initial_state['file_path']}")

@app.post("/v1/process_file_for_dify", response_model=DifyProcessResponse, tags=["Dify Integration"])
async def process_file_for_dify(
    background_tasks: BackgroundTasks,
    dify_dataset_id: str = Form(...),
    file: UploadFile = File(...)
):
    """
    Endpoint ที่ Dify จะเรียกใช้เมื่อมีการอัปโหลดไฟล์
    ระบบจะรับไฟล์, เริ่ม Agentic Pipeline ใน Background,
    และตอบกลับ Dify ทันที
    """
    try:
        # 1. บันทึกไฟล์ที่ Dify ส่งมาลง temp
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            file_path = tmp_file.name
        
        print(f"Received file from Dify for Dataset {dify_dataset_id}. Saved to: {file_path}")

        # 2. เตรียม "ถาด" (State) ใบแรก
        initial_state = {
            "file_path": file_path,
            "original_filename": file.filename,
            "clean_text": "",
            "metadata": {},
            "chunks": [],
            "error_message": None,
            "layout_map": {},
            "validation_passes": 0,
            "retry_history": [],
            # --- [สำคัญ!] ส่ง ID ของ Dify เข้าไปใน State ---
            "dify_integration_config": {
                "dataset_id": dify_dataset_id
            }
        }

        # 3. สั่งให้ LangGraph เริ่มทำงาน (แบบ Background)
        # นี่คือหัวใจสำคัญ: เราตอบ Dify กลับไปทันที
        # ในขณะที่ Graph ของเรากำลังทำงานอยู่เบื้องหลัง
        background_tasks.add_task(run_graph_in_background, initial_state)

        # 4. ตอบกลับ Dify ทันทีว่า "ได้รับเรื่องแล้ว"
        return DifyProcessResponse(
            status="processing_started",
            message="Agentic RAG Pipeline has started processing the file.",
            filename=file.filename
        )
    except Exception as e:
        return DifyProcessResponse(
            status="error",
            message=f"Failed to start processing: {e}",
            filename=file.filename
        )

# --- ส่วนสำหรับรัน Server ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)