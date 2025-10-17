# agentic_rag_pipeline/components/document_preprocessor.py

import os
import re
import io
import docx
import ftfy
import pandas as pd
from pdf2image import convert_from_path
from openai import OpenAI
from langchain.prompts import PromptTemplate

# --- ส่วนประกอบภายใน Component ---
# Import การตั้งค่ากลางและ LLM Provider ของโปรเจกต์เรา
from agentic_rag_pipeline import config
from agentic_rag_pipeline.core.llm_provider import get_llm

# --- 1. OCR Agent (ดัดแปลงจาก layout_analyzer.py) ---
# สร้าง Client ไว้ล่วงหน้าเพื่อประสิทธิภาพที่ดีกว่า
_ocr_client = OpenAI(
    api_key=config.OCR_API_KEY,
    base_url=config.OCR_API_BASE,
    timeout=360.0,
)

def _ocr_image(image_object) -> str:
    """
    รับ Object รูปภาพ แล้วส่งไปให้ OCR service เพื่อสกัดข้อความ
    (Helper function ภายใน ไม่ได้มีไว้ให้เรียกจากข้างนอกโดยตรง)
    """
    try:
        # typhoon-ocr-utils's image_to_base64png is assumed to be available
        # If not, a local implementation might be needed.
        # For now, let's assume it's part of the environment setup.
        from typhoon_ocr.ocr_utils import image_to_base64png
        image_base64 = image_to_base64png(image_object)

        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Return the markdown representation of this document."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
            ],
        }]

        response = _ocr_client.chat.completions.create(
            model="typhoon-ocr-preview",
            messages=messages,
            max_tokens=4096,
        )
        raw_output = response.choices[0].message.content

        # Extract text from the "natural_text" field if present
        match = re.search(r'\{\s*"natural_text":\s*"(.*)"\s*\}', raw_output, re.DOTALL)
        if match:
            return match.group(1).encode('utf-8').decode('unicode_escape')
        return raw_output

    except Exception as e:
        print(f" -> ERROR: เกิดข้อผิดพลาดในการเรียก OCR API: {e}")
        return ""

def _handle_pdf_extraction(file_path: str) -> str:
    """จัดการสกัดข้อความจาก PDF ด้วย OCR"""
    print(" -> ตรวจพบ PDF, เริ่มกระบวนการสกัดด้วย OCR...")
    full_content = []
    try:
        images = convert_from_path(file_path)
        for i, image in enumerate(images):
            print(f" -> กำลัง OCR หน้าที่ {i + 1}/{len(images)}...")
            text_from_page = _ocr_image(image)
            full_content.append(text_from_page)
        return "\\n\\n--- PAGE BREAK ---\\n\\n".join(full_content)
    except Exception as e:
        print(f" -> ERROR: ไม่สามารถแปลง PDF เป็นรูปภาพได้: {e}")
        return ""

# --- 2. Extraction Agent (ดัดแปลงจาก extractor.py) ---
def _extract_raw_text_from_file(file_path: str) -> str:
    """
    ตรวจสอบนามสกุลไฟล์และเลือกวิธีสกัดข้อความดิบที่เหมาะสม
    """
    print(f"สถานีที่ 1.1: กำลังสกัดข้อความดิบจาก {os.path.basename(file_path)}...")
    _, file_extension = os.path.splitext(file_path)
    content = ""
    try:
        if file_extension.lower() == '.pdf':
            content = _handle_pdf_extraction(file_path)
        elif file_extension.lower() == '.docx':
            doc = docx.Document(file_path)
            content = "\\n".join([para.text for para in doc.paragraphs])
        elif file_extension.lower() == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        else:
            print(f" -> WARNING: ไม่รองรับนามสกุลไฟล์: {file_extension}")
            return ""
        
        # ใช้ ftfy ซ่อม "ภาษาต่างดาว" เบื้องต้นเสมอ
        return ftfy.fix_text(content)
        
    except Exception as e:
        print(f" -> ERROR: เกิดข้อผิดพลาดในการสกัดข้อความ: {e}")
        return ""

# --- 3. Proofreader Agent (ดัดแปลงจาก proofreader.py) ---
_proofread_prompt_template = PromptTemplate.from_template(
    """คุณคือบรรณาธิการตรวจทานอักษรที่มีความแม่นยำสูงสุด ภารกิจของคุณมีเพียงหนึ่งเดียวคือการแก้ไขข้อความที่ผิดเพี้ยนจากการสแกน (OCR) หรือการสะกดผิดเล็กน้อย ให้กลับมาเป็นภาษาไทยที่ถูกต้อง

**กฎเหล็กที่คุณต้องปฏิบัติตาม:**
1.  **ห้ามสรุปความ:** ห้ามย่อหรือตัดทอนเนื้อหาใดๆ ทั้งสิ้น
2.  **ห้ามเพิ่มเติม:** ห้ามเพิ่มคำ, ประโยค, หรือข้อมูลที่ไม่มีอยู่ในต้นฉบับ
3.  **รักษารูปแบบเดิม:** คงการขึ้นบรรทัดใหม่และย่อหน้าของเอกสารต้นฉบับไว้ให้เหมือนเดิมที่สุด
4.  **แก้ไขเฉพาะที่ผิด:** จงแก้ไขเฉพาะคำที่สะกดผิด, สระหรือวรรณยุกต์เพี้ยน, หรือตัวอักษรที่ผิดพลาดจากการ OCR อย่างชัดเจนเท่านั้น

**จงส่งคืนเฉพาะข้อความที่พิสูจน์อักษรแล้วเท่านั้น โดยไม่มีคำอธิบายอื่นใดเพิ่มเติม**

**ข้อความต้นฉบับ:**
"{text_to_proofread}"
"""
)

def _convert_html_tables_to_markdown(text: str) -> str:
    """ค้นหาและแปลงตาราง HTML ในข้อความเป็น Markdown"""
    tables = list(re.finditer(r'(<table.*?>.*?</table>)', text, re.DOTALL))
    if not tables:
        return text

    print(f" -> ตรวจพบ {len(tables)} ตาราง HTML, กำลังแปลงเป็น Markdown...")
    # วนลูปจากหลังมาหน้าเพื่อไม่ให้ index เพี้ยนตอนแทนที่
    for table_match in reversed(tables):
        html_table_str = table_match.group(1)
        try:
            # ใช้ pandas อ่าน HTML ที่เป็นตาราง
            df_list = pd.read_html(io.StringIO(html_table_str))
            if df_list:
                markdown_table = df_list[0].to_markdown(index=False)
                start, end = table_match.span()
                text = f"{text[:start]}\\n\\n{markdown_table}\\n\\n{text[end:]}"
        except Exception as e:
            # ถ้าแปลงไม่ได้ก็ข้ามไป ไม่ทำให้โปรแกรมหยุด
            print(f" -> WARNING: ไม่สามารถแปลงตารางได้, ข้ามไป... Error: {e}")
            pass
    return text

def _proofread_text(text: str, llm) -> str:
    """
    ทำความสะอาดและพิสูจน์อักษรข้อความด้วย LLM
    """
    if not text or not text.strip():
        return ""
    
    print("สถานีที่ 1.2: กำลังทำความสะอาดและพิสูจน์อักษร...")
    
    # 1. จัดการ/แปลงรูปแบบเบื้องต้น
    cleaned_text = _convert_html_tables_to_markdown(text)
    cleaned_text = cleaned_text.replace('--- PAGE BREAK ---', '')
    cleaned_text = re.sub(r'\\n{3,}', '\\n\\n', cleaned_text) # ลดการเว้นบรรทัดเกิน

    # 2. พิสูจน์อักษรด้วย LLM (แบ่งส่งทีละส่วนเพื่อไม่ให้ context ยาวเกินไป)
    print(" -> กำลังส่งข้อความให้ LLM ช่วยพิสูจน์อักษร...")
    final_proofread_text = []
    # แบ่งข้อความทุกๆ 4000 ตัวอักษร
    text_parts = [cleaned_text[i:i+4000] for i in range(0, len(cleaned_text), 4000)]

    for i, part in enumerate(text_parts):
        print(f" -> กำลังพิสูจน์อักษรส่วนที่ {i+1}/{len(text_parts)}...")
        formatted_prompt = _proofread_prompt_template.format(text_to_proofread=part)
        response = llm.complete(formatted_prompt)
        final_proofread_text.append(response.text)

    return "".join(final_proofread_text)

# --- 4. Main Function ของ Component ---
def process_document(file_path: str) -> str:
    """
    ฟังก์ชันหลักสำหรับ Component นี้
    รับเส้นทางไฟล์เข้ามา และคืนค่าเป็นข้อความที่ผ่านการสกัดและพิสูจน์อักษรแล้ว
    
    Args:
        file_path (str): The full path to the document file.

    Returns:
        str: The cleaned and proofread text content, or an empty string if processing fails.
    """
    # ตรวจสอบว่าไฟล์มีอยู่จริงหรือไม่
    if not os.path.exists(file_path):
        print(f"ERROR: ไม่พบไฟล์ที่ '{file_path}'")
        return ""
        
    # 1. สกัดข้อความดิบ
    raw_text = _extract_raw_text_from_file(file_path)
    if not raw_text:
        print(f" -> การสกัดข้อความล้มเหลวสำหรับไฟล์ {os.path.basename(file_path)}")
        return ""
    
    # 2. พิสูจน์อักษรข้อความดิบ
    # โหลด LLM ผ่าน provider ของเรา
    llm = get_llm() 
    clean_text = _proofread_text(raw_text, llm)
    
    print(f"✅ Pre-processing สำหรับไฟล์ {os.path.basename(file_path)} เสร็จสิ้น!")
    return clean_text