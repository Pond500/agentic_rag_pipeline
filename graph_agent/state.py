# agentic_rag_pipeline/graph_agent/state.py

from typing import List, Dict, Any, TypedDict

# --- นี่คือโครงสร้าง "ถาด" ของเรา ---
class GraphState(TypedDict):
    """
    เป็น "ถาด" ที่ใช้เก็บสถานะของงานทั้งหมดใน Pipeline ของเรา

    Attributes:
        file_path (str): เส้นทางเต็มของไฟล์ที่กำลังประมวลผล
        original_filename (str): ชื่อไฟล์ดั้งเดิม
        clean_text (str): เนื้อหาที่ผ่านการพิสูจน์อักษรแล้ว
        metadata (dict): Metadata ที่สร้างโดย Librarian Agent
        chunks (list): รายการ Chunks ที่แบ่งเสร็จแล้ว
        error_message (str | None): เก็บข้อความ Error หากมีข้อผิดพลาดเกิดขึ้น

        # --- Fields สำหรับการ "คิดทบทวน" ---
        validation_passes (int): จำนวนครั้งที่การตรวจสอบคุณภาพผ่าน
        chunking_retries (int): จำนวนครั้งที่พยายามแบ่ง Chunk ใหม่
    """
    file_path: str
    original_filename: str
    clean_text: str
    metadata: Dict[str, Any]
    chunks: List[Dict[str, Any]]
    error_message: str | None
    chunking_strategy: str
    chunking_params: Dict[str, Any]
    # สำหรับการวนลูป
    validation_passes: int
    chunking_retries: int