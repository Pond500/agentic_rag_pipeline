# agentic_rag_pipeline/core/agent_orchestrator.py

import os

# --- Import "เครื่องมือ" ทั้งหมดจากคลังของเรา ---
from agentic_rag_pipeline.components import document_preprocessor
from agentic_rag_pipeline.components import metadata_generator
from agentic_rag_pipeline.components import chunker
from agentic_rag_pipeline.components import indexer

def run_full_pipeline_for_file(file_path: str):
    """
    ควบคุมสายพานการผลิตทั้งหมดสำหรับไฟล์เดียว
    นี่คือ "สมอง" หลักของ Agent ที่เรียกใช้เครื่องมือแต่ละชิ้นตามลำดับ

    Args:
        file_path (str): The full path to the document to be processed.
    """
    print(f"\\n{'='*20} เริ่มต้น Pipeline สำหรับไฟล์: {os.path.basename(file_path)} {'='*20}")
    
    # --- สถานีที่ 1: Pre-processing ---
    # แปลงไฟล์ -> Clean Text
    clean_text = document_preprocessor.process_document(file_path)
    if not clean_text:
        print(f"!!! Pipeline หยุดทำงานสำหรับไฟล์นี้เนื่องจากไม่สามารถประมวลผลเอกสารเบื้องต้นได้ !!!")
        return

    # --- สถานีที่ 2: Metadata Generation ---
    # Clean Text -> Structured Metadata
    original_filename = os.path.basename(file_path)
    metadata = metadata_generator.generate_metadata_for_text(clean_text, original_filename)
    if not metadata:
        print(f"!!! Pipeline หยุดทำงานสำหรับไฟล์นี้เนื่องจากไม่สามารถสร้าง Metadata ได้ !!!")
        return
        
    # --- สถานีที่ 3: Chunking ---
    # Clean Text + Metadata -> Smart Chunks
    chunks = chunker.create_chunks_for_text(clean_text, metadata, original_filename)
    if not chunks:
        print(f"!!! Pipeline หยุดทำงานสำหรับไฟล์นี้เนื่องจากไม่สามารถสร้าง Chunks ได้ !!!")
        return

    # --- สถานีที่ 4: Indexing ---
    # Chunks -> Save to Database
    success = indexer.index_document_and_chunks(clean_text, metadata, chunks, original_filename)
    
    if success:
        print(f"\\n{'='*20} Pipeline สำหรับไฟล์ {original_filename} เสร็จสิ้นสมบูรณ์! {'='*20}")
    else:
        print(f"\\n!!! Pipeline สำหรับไฟล์ {original_filename} พบข้อผิดพลาดร้ายแรงในขั้นตอนสุดท้าย !!!")