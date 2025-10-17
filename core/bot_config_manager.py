# src/core/bot_config_manager.py

import psycopg2
from pydantic import BaseModel
from functools import lru_cache

# Import config ของ Dopa_project เพื่อใช้เชื่อมต่อ DB
from src import config

# สร้าง Model สำหรับเก็บข้อมูล Config ของบอท
class BotConfig(BaseModel):
    bot_id: int
    bot_name: str
    qdrant_collection_name: str
    api_key: str
    # Prompts
    system_prompt: str
    routing_prompt: str
    refusal_message: str
    about_bot_message: str

# ใช้ lru_cache เพื่อ "จำ" config ของบอทที่เคยเรียกใช้แล้ว จะได้ไม่ต้อง query DB ทุกครั้ง
# ทำให้ API เร็วขึ้นมาก!
@lru_cache(maxsize=128)
def get_bot_config_by_api_key(api_key: str) -> BotConfig | None:
    """
    ค้นหาข้อมูล Bot และ Persona จาก PostgreSQL โดยใช้ API Key.
    Returns a BotConfig object or None if not found.
    """
    print(f"--- Loading Bot Config for API Key: ...{api_key[-4:]}")
    conn = None
    try:
        # เชื่อมต่อไปยัง DB ที่เก็บ Config (ตอนนี้คือ DB ของ agentic_pipeline)
        conn = psycopg2.connect(
            dbname=config.DB_NAME,
            user=config.DB_USER,
            password=config.DB_PASS,
            host=config.DB_HOST, # ตรวจสอบให้แน่ใจว่า .env ชี้ไปที่ DB ของ agentic_pipeline
            port=config.DB_PORT
        )
        with conn.cursor() as cur:
            # JOIN ตาราง bots และ bot_personas เข้าด้วยกัน
            cur.execute("""
                SELECT
                    b.id,
                    b.bot_name,
                    b.qdrant_collection_name,
                    b.api_key,
                    p.system_prompt,
                    p.routing_prompt,
                    p.refusal_message,
                    p.about_bot_message
                FROM bots b
                JOIN bot_personas p ON b.persona_id = p.id
                WHERE b.api_key = %s AND b.is_active = true;
            """, (api_key,))

            result = cur.fetchone()

            if result:
                return BotConfig(
                    bot_id=result[0],
                    bot_name=result[1],
                    qdrant_collection_name=result[2],
                    api_key=result[3],
                    system_prompt=result[4],
                    routing_prompt=result[5],
                    refusal_message=result[6],
                    about_bot_message=result[7]
                )
        return None
    except Exception as e:
        print(f"!!! DATABASE ERROR while fetching bot config: {e}")
        return None
    finally:
        if conn:
            conn.close()