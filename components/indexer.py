# agentic_rag_pipeline/components/indexer.py

import psycopg2
import json
from typing import List, Dict, Any

# --- Import ส่วนประกอบกลางของโปรเจกต์ ---
from agentic_rag_pipeline import config
from agentic_rag_pipeline.core.llm_provider import get_embed_model

# --- 1. Helper Function สำหรับเชื่อมต่อ Database ---

def _get_db_connection():
    """
    สร้างและคืนค่า connection ไปยัง PostgreSQL โดยใช้ค่าจาก config กลาง
    """
    try:
        conn = psycopg2.connect(
            dbname=config.DB_NAME,
            user=config.DB_USER,
            password=config.DB_PASS,
            host=config.DB_HOST,
            port=config.DB_PORT
        )
        return conn
    except psycopg2.OperationalError as e:
        print(f" -> ERROR: การเชื่อมต่อฐานข้อมูลล้มเหลว: {e}")
        return None

# --- 2. Main Function ของ Component ---

def index_document_and_chunks(
    full_text: str,
    metadata: Dict[str, Any],
    chunks: List[Dict[str, Any]],
    original_filename: str
) -> bool:
    """
    ฟังก์ชันหลักสำหรับ Component นี้ (Agent Indexer)
    รับข้อมูลทั้งหมดของเอกสาร, สร้าง Embedding, และบันทึกลงฐานข้อมูล
    ทั้งตาราง `knowledge_items` (เอกสารหลัก) และ `knowledge_chunks` (หน่วยข้อมูลย่อย)

    Args:
        full_text (str): เนื้อหาทั้งหมดของเอกสาร (Clean Text)
        metadata (Dict[str, Any]): Metadata ที่สร้างโดย Librarian Agent
        chunks (List[Dict[str, Any]]): List ของ Chunks ที่สร้างโดย Chunker Agent
        original_filename (str): ชื่อไฟล์ดั้งเดิม

    Returns:
        bool: True ถ้าการบันทึกสำเร็จ, False ถ้าล้มเหลว
    """
    print("สถานีที่ 4: Agent Indexer กำลังสร้าง Embedding และบันทึกข้อมูล...")

    if not chunks:
        print(" -> WARNING: ไม่มี Chunks ให้บันทึก ข้ามการทำงาน")
        return False

    conn = _get_db_connection()
    if not conn:
        return False

    item_id = None
    try:
        with conn.cursor() as cur:
            # --- ขั้นตอนที่ 1: บันทึกเอกสารหลัก (Parent Document) ลงใน knowledge_items ---
            print(f" -> กำลังบันทึกเอกสารหลัก '{metadata.get('document_title', original_filename)}'")
            
            # เตรียม Metadata สำหรับตาราง knowledge_items
            item_metadata = {
                "type": "RAG",
                "original_filename": original_filename,
                "category": metadata.get("document_type", "อื่นๆ"),
                "tags": metadata.get("main_topics", [])
            }

            cur.execute(
                """
                INSERT INTO knowledge_items (source_type, status, title, full_content, metadata)
                VALUES (%s, %s, %s, %s, %s) RETURNING id;
                """,
                ('RAG', 'active', metadata.get('document_title'), full_text, json.dumps(item_metadata, ensure_ascii=False))
            )
            item_id = cur.fetchone()[0]
            print(f" -> บันทึกเอกสารหลักสำเร็จ ได้รับ ID: {item_id}")

            # --- ขั้นตอนที่ 2: สร้าง Embeddings สำหรับทุก Chunks ---
            embed_model = get_embed_model()
            
            texts_to_embed = [chunk['content'] for chunk in chunks]
            print(f" -> กำลังสร้าง Embeddings สำหรับ {len(texts_to_embed)} Chunks...")
            
            embeddings = embed_model.encode(texts_to_embed, normalize_embeddings=True)
            print(" -> สร้าง Embeddings สำเร็จ")

            # --- ขั้นตอนที่ 3: บันทึกแต่ละ Chunk ลงใน knowledge_chunks ---
            print(f" -> กำลังบันทึก Chunks ทั้ง {len(chunks)} ชิ้นลงฐานข้อมูล...")
            for i, chunk in enumerate(chunks):
                chunk_metadata = chunk['metadata']
                chunk_metadata['knowledge_item_id'] = item_id # <<-- เชื่อมโยงกลับไปยังเอกสารหลัก
                
                embedding_vector = embeddings[i].tolist()

                cur.execute(
                    """
                    INSERT INTO knowledge_chunks (knowledge_item_id, chunk_text, chunk_sequence, embedding, metadata)
                    VALUES (%s, %s, %s, %s, %s);
                    """,
                    (
                        item_id,
                        chunk['content'],
                        chunk_metadata.get('chunk_number', i + 1),
                        json.dumps(embedding_vector), # แปลง list เป็น JSON string
                        json.dumps(chunk_metadata, ensure_ascii=False)
                    )
                )

        # ถ้าทุกอย่างสำเร็จ ให้ commit transaction
        conn.commit()
        print(f"✅ Indexing สำหรับไฟล์ {original_filename} เสร็จสิ้นสมบูรณ์!")
        return True

    except Exception as e:
        print(f" -> ERROR: เกิดข้อผิดพลาดร้ายแรงระหว่างการ Indexing: {e}")
        if conn:
            conn.rollback() # ย้อนกลับการเปลี่ยนแปลงทั้งหมดถ้ามีข้อผิดพลาด
        return False
    finally:
        if conn:
            conn.close()