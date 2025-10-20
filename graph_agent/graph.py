# agentic_rag_pipeline/graph_agent/graph.py

from langgraph.graph import StateGraph, END
from typing import Literal

# --- Import "ถาด" และ "สถานีทำงาน" ทั้งหมดของเรา ---
from .state import GraphState
from .nodes import (
    preprocess_node,
    metadata_node,
    chunker_node,
    strategize_chunking_node,
    validate_chunks_node,
    index_node
)

# ==============================================================================
# 1. สร้าง "ทางแยก" (Conditional Edge)
# ==============================================================================
def should_continue(state: GraphState) -> Literal["continue", "retry_chunking", "end"]:
    """
    นี่คือ "สมอง" ของทางแยก ที่จะตัดสินใจว่าจะไปทางไหนต่อ
    หลังจากสถานี Validate Chunks ทำงานเสร็จ
    """
    print("--- 🚦 ทางแยก: Deciding next step ---")

    # กรณีที่ 1: เกิด Error ร้ายแรงขึ้นระหว่างทาง -> หยุดทำงานทันที
    if state.get("error_message"):
        print("   -> 🛑 Decision: พบ Error, หยุดการทำงาน")
        return "end"

    # กรณีที่ 2: การตรวจสอบคุณภาพผ่าน -> ไปต่อยังสถานีถัดไป
    if state.get("validation_passes", 0) > 0:
        print("   -> ✅ Decision: คุณภาพผ่าน, ไปยังสถานี Index")
        return "continue"

    retry_count = len(state.get("retry_history", []))
    if retry_count < 5: # ลองซ้ำไม่เกิน 3 ครั้ง
        print(f"   -> 🔄 Decision: คุณภาพไม่ผ่าน, วนกลับไปทำ Chunker ใหม่ (ครั้งที่ {retry_count + 1})")
        return "retry_chunking"

    # กรณีที่ 4: ลองซ้ำครบ 3 ครั้งแล้วยังไม่ผ่าน -> ยอมแพ้และหยุดทำงาน
    else:
        print("   -> 🛑 Decision: ลองทำ Chunker ซ้ำครบ 3 ครั้งแล้วยังล้มเหลว, หยุดการทำงาน")
        state['error_message'] = "Chunking validation failed after multiple retries."
        return "end"

# ==============================================================================
# 2. สร้าง "พิมพ์เขียวโรงงาน" (The Graph Definition)
# ==============================================================================
def create_graph():
    workflow = StateGraph(GraphState)

    # --- เพิ่ม "สถานีทำงาน" ใหม่เข้าไป ---
    workflow.add_node("preprocess", preprocess_node)
    workflow.add_node("generate_metadata", metadata_node)
    workflow.add_node("strategize_chunking", strategize_chunking_node) # <--- เพิ่มสถานีใหม่
    workflow.add_node("chunker", chunker_node)
    workflow.add_node("validate_chunks", validate_chunks_node)
    workflow.add_node("index", index_node)

    workflow.set_entry_point("preprocess")

    # --- [ใหม่!] เดินสายพานใหม่ ---
    workflow.add_edge("preprocess", "generate_metadata")
    workflow.add_edge("generate_metadata", "strategize_chunking") # <-- จาก metadata ไปหา strategist
    workflow.add_edge("strategize_chunking", "chunker")       # <-- จาก strategist ไปหา chunker
    workflow.add_edge("chunker", "validate_chunks")

    # --- ทางแยกยังคงทำงานเหมือนเดิม ---
    workflow.add_conditional_edges(
        "validate_chunks",
        should_continue,
        {
            "continue": "index",
            "retry_chunking": "chunker", # วนกลับไปที่ chunker เหมือนเดิม
            "end": END
        }
    )

    workflow.add_edge("index", END)

    app = workflow.compile()
    return app

graph_app = create_graph()