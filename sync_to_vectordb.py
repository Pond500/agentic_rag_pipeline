# agentic_rag_pipeline/sync_to_vectordb.py

import psycopg2
import json
from qdrant_client import models, QdrantClient

# --- Import ส่วนประกอบจากโปรเจกต์ของเรา ---
from agentic_rag_pipeline import config
from agentic_rag_pipeline.core.llm_provider import get_embed_model

def get_source_db_connection():
    """เชื่อมต่อฐานข้อมูลต้นทาง (PostgreSQL ของ Agent)"""
    try:
        conn = psycopg2.connect(
            dbname=config.DB_NAME,
            user=config.DB_USER,
            password=config.DB_PASS,
            host=config.DB_HOST,
            port=config.DB_PORT
        )
        print("✅ เชื่อมต่อฐานข้อมูล PostgreSQL (ต้นทาง) สำเร็จ")
        return conn
    except psycopg2.OperationalError as e:
        print(f"❌ การเชื่อมต่อ PostgreSQL ล้มเหลว: {e}")
        return None

def get_destination_qdrant_client():
    """เชื่อมต่อ Vector DB ปลายทาง (Qdrant ของ Agent Pipeline)"""
    try:
        # แก้ไขให้ใช้ Config ใหม่สำหรับ Agent Qdrant
        client = QdrantClient(host=config.AGENT_QDRANT_HOST, port=config.AGENT_QDRANT_PORT)
        print("✅ เชื่อมต่อ Qdrant (Agent Pipeline) สำเร็จ")
        return client
    except Exception as e:
        print(f"❌ การเชื่อมต่อ Qdrant ล้มเหลว: {e}")
        return None

def main():
    print("\n--- 🚀 เริ่มกระบวนการ Sync ข้อมูลจาก PostgreSQL ไปยัง Qdrant (Agent's DB) ---")

    conn = get_source_db_connection()
    qdrant_client = get_destination_qdrant_client()
    embed_model = get_embed_model() # โหลด Embedding model เพื่อเอาขนาดของ Vector

    if not conn or not qdrant_client or not embed_model:
        print("!!! ไม่สามารถเริ่มกระบวนการได้เนื่องจากการเชื่อมต่อล้มเหลว !!!")
        return

    # แก้ไขให้ใช้ Config ใหม่สำหรับ Agent Qdrant
    collection_name = config.AGENT_QDRANT_COLLECTION_NAME
    # ดึงขนาด vector จาก model ที่เราใช้ใน pipeline
    vector_size = embed_model.get_sentence_embedding_dimension()

    # --- 1. อ่านข้อมูล Chunks ทั้งหมดจาก PostgreSQL ---
    points_to_upsert = []
    try:
        with conn.cursor() as cur:
            print(f"\n1. กำลังดึงข้อมูล Chunks ทั้งหมดจาก PostgreSQL (DB: {config.DB_NAME})...")
            # แก้ไข Query ให้ดึงเฉพาะ id, chunk_text, embedding, และ metadata ที่จำเป็น
            cur.execute("SELECT id, chunk_text, embedding, metadata FROM knowledge_chunks")
            
            for row in cur.fetchall():
                chunk_id, chunk_text, embedding_str, metadata = row
                
                if not embedding_str or not metadata:
                    print(f"  > ข้าม Chunk ID: {chunk_id} เนื่องจากข้อมูลไม่สมบูรณ์")
                    continue
                
                # Payload ใน Qdrant จะใช้ metadata ที่เราสร้างจาก Librarian Agent
                # และเพิ่ม 'text' เข้าไปเพื่อให้สามารถทำ Keyword Search ได้
                payload = metadata.copy()
                payload['text'] = chunk_text

                points_to_upsert.append(
                    models.PointStruct(
                        id=chunk_id,
                        vector=json.loads(embedding_str), # แปลง JSON string กลับเป็น list
                        payload=payload
                    )
                )
        print(f"   -> ดึงข้อมูลมาได้ทั้งหมด {len(points_to_upsert)} Chunks")
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาดในการอ่านข้อมูลจาก PostgreSQL: {e}")
        return
    finally:
        if conn:
            conn.close()

    if not points_to_upsert:
        print("ไม่พบข้อมูลที่จะย้ายไปยัง Qdrant.")
        return

    # --- 2. ลบและสร้าง Collection ใหม่ใน Qdrant เสมอเพื่อให้ข้อมูลสดใหม่ ---
    try:
        print(f"\n2. กำลังลบและสร้าง Collection ใหม่ใน Qdrant: '{collection_name}'...")
        qdrant_client.recreate_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        )
        print("   -> สร้าง Collection สำเร็จ!")
    except Exception as e:
        print(f"❌ ไม่สามารถตั้งค่า Collection ใน Qdrant ได้: {e}")
        return

    # --- 3. Upsert ข้อมูลทั้งหมดเข้า Qdrant ---
    print(f"\n3. กำลังนำเข้า {len(points_to_upsert)} points ไปยัง Qdrant...")
    qdrant_client.upsert(
        collection_name=collection_name,
        points=points_to_upsert,
        wait=True,
    )
    
    print("\n--- 🎉 Sync ข้อมูลไปยัง Qdrant สำเร็จ! ---")
    
    # --- 4. ตรวจสอบผลลัพธ์ ---
    collection_info = qdrant_client.get_collection(collection_name=collection_name)
    print("\n--- 📊 สรุปผลลัพธ์ใน Qdrant ---")
    print(f"   ชื่อ Collection: {collection_name}")
    print(f"   จำนวน Points ทั้งหมด: {collection_info.points_count}")

if __name__ == "__main__":
    main()