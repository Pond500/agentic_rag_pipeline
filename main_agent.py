# agentic_rag_pipeline/main_agent.py

import os
import argparse

# --- Import ส่วนประกอบหลัก ---
from . import config
from .core import agent_orchestrator

def find_documents_to_process(root_folder: str) -> list[str]:
    """
    ค้นหาไฟล์เอกสารทั้งหมด (.pdf, .docx, .txt) ในโฟลเดอร์ที่กำหนดและโฟลเดอร์ย่อยทั้งหมด
    """
    supported_extensions = ('.pdf', '.docx', '.txt')
    filepaths = []
    print(f"--- กำลังค้นหาเอกสารใน: {root_folder} ---")
    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.lower().endswith(supported_extensions):
                filepaths.append(os.path.join(dirpath, filename))
    print(f"--- พบเอกสารทั้งหมด {len(filepaths)} ไฟล์ ---")
    return filepaths

def main():
    """
    Main entry point for the Agentic RAG Pipeline.
    """
    print("🚀 Agentic RAG Pipeline กำลังเริ่มต้น...")

    # --- ตั้งค่าการรับ Argument จาก Command Line (เผื่ออนาคต) ---
    parser = argparse.ArgumentParser(description="Run the Agentic RAG Pipeline.")
    parser.add_argument(
        '--path',
        type=str,
        default=config.DATA_ROOT_FOLDER,
        help=f"Path to the folder containing documents. Defaults to DATA_ROOT_FOLDER in config ('{config.DATA_ROOT_FOLDER}')."
    )
    args = parser.parse_args()

    # ตรวจสอบว่าโฟลเดอร์ข้อมูลมีอยู่จริง
    if not os.path.isdir(args.path):
        print(f"!!! ERROR: ไม่พบโฟลเดอร์ข้อมูลที่ '{args.path}'")
        print("กรุณาสร้างโฟลเดอร์และนำเอกสารไปใส่ หรือระบุ path ที่ถูกต้องด้วย --path")
        return

    # 1. ค้นหาเอกสารทั้งหมดที่ต้องประมวลผล
    files_to_process = find_documents_to_process(args.path)

    if not files_to_process:
        print("ไม่พบเอกสารที่รองรับให้ประมวลผลในโฟลเดอร์ที่กำหนด")
        return

    # 2. วนลูปและสั่งให้ Orchestrator จัดการทีละไฟล์
    for file_path in files_to_process:
        try:
            agent_orchestrator.run_full_pipeline_for_file(file_path)
        except Exception as e:
            print(f"\\n{'!'*20} เกิดข้อผิดพลาดร้ายแรงที่ไม่สามารถจัดการได้กับไฟล์ {os.path.basename(file_path)} {'!'*20}")
            print(f"Error: {e}")
            # การทำงานจะดำเนินต่อไปยังไฟล์ถัดไป

    print("\\n🎉🎉🎉 การประมวลผลเอกสารทั้งหมดเสร็จสิ้น! 🎉🎉🎉")

if __name__ == "__main__":
    main()