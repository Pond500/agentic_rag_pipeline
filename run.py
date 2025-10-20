# agentic_rag_pipeline/run.py

import argparse
import pprint # Library สำหรับพิมพ์ Dictionary สวยๆ

# --- Import "โรงงาน" (Graph) ที่เราสร้างไว้ ---
from agentic_rag_pipeline.graph_agent.graph import graph_app

def main():
    """
    นี่คือ "ปุ่ม Start" ของเรา
    """
    # --- 1. ตั้งค่าการรับ Argument จาก Command Line ---
    parser = argparse.ArgumentParser(
        description="Run the Agentic RAG Pipeline with LangGraph."
    )
    parser.add_argument(
        "file_path",
        type=str,
        help="The full path to the document file you want to process."
    )
    args = parser.parse_args()

    print(f"--- 🚀 Starting Agentic Pipeline for: {args.file_path} ---")

    # --- 2. เตรียม "ถาด" (State) ใบแรกสำหรับส่งเข้าโรงงาน ---
    initial_state = {
        "file_path": args.file_path,
        "original_filename": "",
        "clean_text": "",
        "metadata": {},
        "chunks": [],
        "error_message": None,
        "validation_passes": 0,
        "retry_history": [],
        
    }

    # --- 3. ส่ง "ถาด" เข้าโรงงานและเริ่มทำงาน! ---
    # .invoke() คือคำสั่ง "Start"
    final_state = graph_app.invoke(initial_state)

    # --- 4. แสดงผลลัพธ์สุดท้ายจาก "ถาด" ใบสุดท้าย ---
    print("\n" + "="*50)
    print("--- 🎉 Pipeline Finished! Final State: ---")
    pprint.pprint(final_state)
    print("="*50)


if __name__ == "__main__":
    main()