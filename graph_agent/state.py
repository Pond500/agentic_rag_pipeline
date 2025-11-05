# agentic_rag_pipeline/graph_agent/state.py

from typing import List, Dict, Any, TypedDict

# --- นี่คือโครงสร้าง "ถาด" V5 + V2 + Dify ---
class GraphState(TypedDict):
    """
    เป็น "ถาด" ที่ใช้เก็บสถานะของงานทั้งหมด (V5 + V2)

    Attributes:
        file_path (str): เส้นทางเต็มของไฟล์ที่กำลังประมวลผล
        original_filename (str): ชื่อไฟล์ดั้งเดิม
        clean_text (str): เนื้อหาที่ผ่านการพิสูจน์อักษรแล้ว
        metadata (dict): Metadata ที่สร้างโดย Librarian Agent
        chunks (list): รายการ Chunks ที่แบ่งเสร็จแล้ว
        error_message (str | None): เก็บข้อความ Error หากมีข้อผิดพลาดเกิดขึ้น

        # --- [V2] Fields สำหรับ "นักวิเคราะห์โครงสร้าง" ---
        layout_map: Dict[str, Any]  # <-- [ใหม่!] เก็บแผนผังโครงสร้าง (Layout Map)
        
        # --- [V5] Fields สำหรับ "แพทย์ผู้เชี่ยวชาญ" ---
        validation_passes: int
        retry_history: List[Dict[str, Any]] # <-- "แฟ้มประวัติผู้ป่วย" (เหมือน V4)

        # --- [ใหม่!] ขั้นตอนที่ 1: เพิ่ม Field สำหรับ Dify ---
        dify_integration_config: Dict[str, Any]
    """
    file_path: str
    original_filename: str
    clean_text: str
    metadata: Dict[str, Any]
    chunks: List[Dict[str, Any]]
    error_message: str | None
    
    # --- [V2] ---
    layout_map: Dict[str, Any] 
    
    # --- [V5] ---
    validation_passes: int
    retry_history: List[Dict[str, Any]]

    # --- [ใหม่!] Field สำหรับ Dify ---
    dify_integration_config: Dict[str, Any]