# agentic_rag_pipeline/components/metadata_generator.py

import json
import re
from langchain.prompts import PromptTemplate
from typing import Dict, Any

# Import LLM Provider ของโปรเจกต์เรา
from agentic_rag_pipeline.core.llm_provider import get_llm

# --- 1. Prompt Template (The Brain of the Librarian) ---
# นี่คือ Prompt ที่ดีที่สุดของคุณจาก smart_agent/pipeline/librarian.py
# มันละเอียดและครอบคลุมมาก ทำให้ LLM ทำงานได้ตรงเป้าหมาย
_METADATA_PROMPT = PromptTemplate.from_template(
    """คุณคือบรรณารักษ์ผู้เชี่ยวชาญด้านเอกสารราชการไทย หน้าที่ของคุณคืออ่านเนื้อหาของเอกสารต่อไปนี้ แล้วสร้าง Metadata ที่เป็นประโยชน์และครอบคลุมที่สุดในรูปแบบ JSON เท่านั้น

**กฎเหล็ก:**
- วิเคราะห์จาก "เนื้อหาเอกสาร" ที่ให้มาเท่านั้น
- ตอบกลับเป็น JSON object ที่สมบูรณ์และถูกต้องตามโครงสร้างเท่านั้น ห้ามมีข้อความอื่นใดๆ ปะปนเด็ดขาด

**เนื้อหาเอกสาร (ส่วนต้น):**
"{document_text}"

**จงสร้างผลลัพธ์ในรูปแบบ JSON ที่มีโครงสร้างดังนี้เท่านั้น:**
{{
  "document_title": "สร้างชื่อเอกสารที่เป็นทางการและสื่อความหมายได้ชัดเจนที่สุดจากเนื้อหา (เช่น 'ระเบียบกรมการปกครองว่าด้วยการจัดทำทะเบียนราษฎร พ.ศ. 2535')",
  "document_type": "เลือกประเภทเอกสารที่เหมาะสมที่สุดจาก: 'คู่มือ', 'ระเบียบ', 'แนวคำวินิจฉัย', 'ประกาศ', 'คำถาม-คำตอบ', 'กฎหมาย', 'อื่นๆ'",
  "summary": "สรุปย่อภาพรวมของ 'ทั้งเอกสาร' ให้ได้ใจความสำคัญภายใน 2-3 ประโยค",
  "main_topics": [
    "Keyword หลักที่ 1",
    "Keyword หลักที่ 2",
    "Keyword หลักที่ 3",
    "Keyword หลักที่ 4"
  ],
  "target_audience": "เลือกกลุ่มเป้าหมายหลักของเอกสารนี้จาก: 'ประชาชน', 'เจ้าหน้าที่', 'นิติบุคคล', 'ทั่วไป'",
  "publication_date": "ค้นหา 'วันที่ประกาศใช้เอกสาร' จากเนื้อหา แล้วตอบเป็นรูปแบบ 'YYYY-MM-DD' เท่านั้น หากไม่พบจริงๆ ให้เป็น null"
}}
"""
)

def _parse_json_from_llm_response(raw_text: str) -> Dict[str, Any]:
    """
    พยายามแยก JSON object ออกจากข้อความที่ LLM ตอบกลับมาอย่างสุดความสามารถ
    """
    # วิธีที่ 1: ค้นหา JSON block ที่อยู่ใน ```json ... ```
    match = re.search(r'```json\s*(\{.*?\})\s*```', raw_text, re.DOTALL)
    if match:
        json_string = match.group(1)
        try:
            return json.loads(json_string)
        except json.JSONDecodeError:
            print(" -> WARNING: พบ JSON ใน ``` แต่ไม่สามารถ parse ได้, กำลังลองวิธีต่อไป...")

    # วิธีที่ 2: ค้นหา JSON object แรกที่เจอในข้อความ
    match = re.search(r'\{.*\}', raw_text, re.DOTALL)
    if match:
        json_string = match.group(0)
        try:
            return json.loads(json_string)
        except json.JSONDecodeError as e:
            print(f" -> ERROR: ไม่สามารถ parse JSON ที่สกัดมาได้: {e}")
            return None
    
    print(" -> ERROR: ไม่พบ JSON object ที่ถูกต้องในคำตอบของ LLM")
    return None


# --- 2. Main Function ของ Component ---
def generate_metadata_for_text(text: str, original_filename: str) -> Dict[str, Any]:
    """
    ฟังก์ชันหลักสำหรับ Component นี้
    รับข้อความที่สะอาด (Clean Text) และชื่อไฟล์เข้ามา แล้วใช้ LLM สร้าง Metadata คืนค่าเป็น Dictionary

    Args:
        text (str): The clean text content of the document.
        original_filename (str): The original filename for context.

    Returns:
        Dict[str, Any]: A dictionary containing the generated metadata, 
                        or a basic dictionary with the filename if generation fails.
    """
    print("สถานีที่ 2: Agent บรรณารักษ์กำลังวิเคราะห์เนื้อหา...")
    
    # --- สร้างค่าเริ่มต้นเผื่อกรณีที่ LLM ทำงานผิดพลาด ---
    fallback_metadata = {
        "document_title": original_filename,
        "document_type": "อื่นๆ",
        "summary": "ไม่สามารถสร้างสรุปได้",
        "main_topics": [],
        "target_audience": "ทั่วไป",
        "publication_date": None
    }

    if not text or not text.strip():
        print(" -> ERROR: เนื้อหาว่างเปล่า ไม่สามารถสร้าง Metadata ได้")
        return fallback_metadata
        
    try:
        llm = get_llm()
        # ใช้เนื้อหาแค่ 8000 ตัวอักษรแรกเพื่อประหยัด Token และเวลา
        formatted_prompt = _METADATA_PROMPT.format(document_text=text[:8000])
        
        print(" -> กำลังส่งเนื้อหาให้ LLM ช่วยสร้าง Metadata...")
        response = llm.complete(formatted_prompt)
        raw_response_text = response.text
        
        metadata = _parse_json_from_llm_response(raw_response_text)
        
        if metadata:
            print(" -> สร้าง Metadata สำเร็จ!")
            return metadata
        else:
            print(" -> WARNING: การสร้าง Metadata ล้มเหลว, จะใช้ข้อมูลพื้นฐานแทน")
            return fallback_metadata

    except Exception as e:
        print(f" -> ERROR: เกิดข้อผิดพลาดร้ายแรงในระหว่างการสร้าง Metadata: {e}")
        return fallback_metadata