# agentic_rag_pipeline/graph_agent/graph.py (เวอร์ชัน V5 + V2 + Dify)

from langgraph.graph import StateGraph, END
from typing import Literal

# --- Import "ถาด" และ "สถานีทำงาน" ทั้งหมดของเรา ---
from .state import GraphState
from .nodes import (
    preprocess_node,
    metadata_node,
    chunker_node,
    layout_analysis_node,  # <-- [V2] อัปเดตจาก strategize_chunking_node
    validate_chunks_node,  # (นี่คือ Validator V5)
    # index_node (เราจะไม่ใช้ตัวนี้แล้ว)
    index_to_dify_node     # <-- [ใหม่!] ขั้นตอนที่ 3: Import Node ใหม่
)

# ==============================================================================
# 1. สร้าง "ทางแยก" (Conditional Edge) (เวอร์ชัน V5 - รองรับ 5 ครั้ง)
# ==============================================================================
def should_continue(state: GraphState) -> Literal["continue", "retry_chunking", "end"]:
    """
    นี่คือ "สมอง" ของทางแยก (V5) ที่จะตัดสินใจว่าจะไปทางไหนต่อ
    หลังจากสถานี Validate Chunks ทำงานเสร็จ
    """
    print("--- 🚦 ทางแยก (V5): Deciding next step ---")

    # กรณีที่ 1: เกิด Error ร้ายแรง หรือ Validator สั่ง "GIVE_UP" (V5)
    if state.get("error_message"):
        print(f"   -> 🛑 Decision: พบ Error ('{state.get('error_message')}'), หยุดการทำงาน")
        return "end"

    # กรณีที่ 2: การตรวจสอบคุณภาพผ่าน -> ไปต่อยังสถานีถัดไป
    if state.get("validation_passes", 0) > 0:
        print("   -> ✅ Decision: คุณภาพผ่าน, ไปยังสถานี Index to Dify") # <-- แก้ไข Log
        return "continue"

    # กรณีที่ 3: การตรวจสอบคุณภาพไม่ผ่าน แต่ยังลองซ้ำได้
    retry_count = len(state.get("retry_history", []))
    
    # --- [อัปเดต!] เพิ่มจำนวนครั้งเป็น 5 ตามที่คุณต้องการ ---
    if retry_count < 5: 
        print(f"   -> 🔄 Decision: คุณภาพไม่ผ่าน, วนกลับไปทำ Chunker ใหม่ (ครั้งที่ {retry_count + 1} / 5)")
        return "retry_chunking"

    # กรณีที่ 4: ลองซ้ำครบ 5 ครั้งแล้วยังไม่ผ่าน -> ยอมแพ้และหยุดทำงาน
    else:
        print(f"   -> 🛑 Decision: ลองทำ Chunker ซ้ำครบ 5 ครั้งแล้วยังล้มเหลว, หยุดการทำงาน")
        state['error_message'] = f"Chunking validation failed after {retry_count} retries."
        return "end"

# ==============================================================================
# 2. สร้าง "พิมพ์เขียวโรงงาน" (The Graph Definition) (เวอร์ชัน V2 + Dify)
# ==============================================================================
def create_graph():
    workflow = StateGraph(GraphState)

    # --- เพิ่ม "สถานีทำงาน" (V2) ---
    workflow.add_node("preprocess", preprocess_node)
    workflow.add_node("generate_metadata", metadata_node)
    workflow.add_node("layout_analysis", layout_analysis_node) # <--- [V2] อัปเดต
    workflow.add_node("chunker", chunker_node)
    workflow.add_node("validate_chunks", validate_chunks_node)
    # workflow.add_node("index", index_node) # <-- ไม่ใช้แล้ว
    workflow.add_node("index_to_dify", index_to_dify_node) # <-- [ใหม่!] ขั้นตอนที่ 3: เพิ่ม Node ใหม่

    workflow.set_entry_point("preprocess")

    # --- [V2] เดินสายพานใหม่ ---
    workflow.add_edge("preprocess", "generate_metadata")
    workflow.add_edge("generate_metadata", "layout_analysis") # <-- [V2] จาก metadata ไปหา layout_analysis
    workflow.add_edge("layout_analysis", "chunker")       # <-- [V2] จาก layout_analysis ไปหา chunker
    workflow.add_edge("chunker", "validate_chunks")

    # --- ทางแยก (V5) ทำงานเหมือนเดิม ---
    workflow.add_conditional_edges(
        "validate_chunks",
        should_continue,
        {
            # --- [ใหม่!] ขั้นตอนที่ 3: เปลี่ยน "continue" ให้ชี้ไปที่ Node ใหม่ ---
            "continue": "index_to_dify",
            "retry_chunking": "chunker", # วนกลับไปที่ chunker (V2) ซึ่งจะอ่าน "ยา" (V5) จาก state
            "end": END
        }
    )

    # --- [ใหม่!] ขั้นตอนที่ 3: เปลี่ยนทางออกสุดท้าย ---
    workflow.add_edge("index_to_dify", END)

    app = workflow.compile()
    return app

graph_app = create_graph()