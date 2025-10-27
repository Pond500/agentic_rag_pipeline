# agentic_rag_pipeline/graph_agent/nodes.py (เวอร์ชัน Intelligent Retry)

import os
import json
import requests
from langchain.prompts import PromptTemplate

# --- Import "ถาด" State และ LLM Provider ของเรา ---
from .state import GraphState
from agentic_rag_pipeline.core.llm_provider import get_llm

# --- API Server URL ---
API_BASE_URL = "http://localhost:8001"

# --- Prompt สำหรับ Validator LLM ---
ULTIMATE_VALIDATION_PROMPT_V5 = PromptTemplate.from_template(
    """คุณคือ "แพทย์ผู้เชี่ยวชาญด้านการแบ่งข้อมูล AI" (V5) ภารกิจของคุณคือการตรวจสอบคุณภาพของ "Chunk ปัจจุบัน"

---
### **ข้อมูลคนไข้ (Chunk ปัจจุบัน):**
1.  **หัวข้อหลัก:** "{document_title}"
2.  **Chunk ก่อนหน้า:** ... {previous_chunk_text} ...
3.  **Chunk ปัจจุบัน (ที่ต้องตรวจ):** ... {current_chunk_text} ...
4.  **Chunk นี้มาจาก "ส่วน" (Section):** (ข้อมูลจากแผนผังโครงสร้าง)
    - Section ID: {section_id}
    - Section Title: "{section_title}"
    - Strategy ที่ใช้: "{strategy_used}"

---
### **แฟ้มประวัติการรักษา (ที่ล้มเหลว):**
(นี่คือสิ่งที่เคยลองทำไปแล้ว และผลลัพธ์คือ "ไม่ผ่าน")
---
{retry_history_str}
---

### **ภารกิจการตรวจสอบ (Your Mission):**

จงวิเคราะห์ข้อมูลทั้งหมด และประเมินคุณภาพของ **"Chunk ปัจจุบัน"**

**หากคุณภาพผ่าน (is_valid: true):**
ให้ตอบกลับ JSON ที่มี "is_valid" เป็น true เท่านั้น

**หากคุณภาพไม่ผ่าน (is_valid: false):**
1.  วินิจฉัย (diagnose) ปัญหา
2.  อ่าน "แฟ้มประวัติ" และ "ข้อมูลส่วน" (Section)
3.  สั่งยา (recommendation) เพื่อแก้ปัญหาที่ "ส่วน" (Section) นี้
    - "action": เลือก "RETRY_SECTION" (ลองใหม่ที่ส่วนนี้) หรือ "GIVE_UP" (ยอมแพ้)
    - "suggestion": (ถ้า action=RETRY_SECTION) แนะนำกลยุทธ์ใหม่สำหรับ "ส่วน" นี้ (เช่น 'semantic', 'structural')

**จงตอบกลับเป็น JSON object ที่มีโครงสร้างดังนี้เท่านั้น:**
{{
  "is_valid": boolean,
  "reason": "อธิบายเหตุผลที่ 'ไม่ผ่าน'",
  "diagnose": "วิเคราะห์สาเหตุ",
  "recommendation": {{
    "action": "RETRY_SECTION | GIVE_UP",
    "target_section_id": {section_id},
    "suggestion": "(ถ้า RETRY_SECTION) 'semantic' | 'structural' | 'recursive'"
  }}
}}

**ตัวอย่างผลลัพธ์ (กรณีสั่งยา):**
{{
  "is_valid": false,
  "reason": "ประโยคถูกตัดจบกลางคัน",
  "diagnose": "กลยุทธ์ 'recursive' ที่ใช้กับ Section {section_id} ('{section_title}') ไม่เหมาะสม",
  "recommendation": {{
    "action": "RETRY_SECTION",
    "target_section_id": {section_id},
    "suggestion": "semantic"
  }}
}}
"""
)

LAYOUT_ANALYSIS_PROMPT_V2 = PromptTemplate.from_template(
    """คุณคือ "สถาปนิกโครงสร้างเอกสาร" (Document Structure Architect) ภารกิจของคุณคือการสแกน "เนื้อหาเอกสารทั้งฉบับ" แล้วแบ่งมันออกเป็น "ส่วน" (Sections) ตามโครงสร้างหรือหัวข้อที่ชัดเจน

[กฎเหล็ก]
1.  วิเคราะห์จาก "เนื้อหาทั้งฉบับ" ({document_text})
2.  แบ่งเอกสารออกเป็น "ส่วน" (Sections) ตามตรรกะของเนื้อหา (เช่น บทนำ, บทที่ 1, ภาคผนวก)
3.  สำหรับแต่ละ "ส่วน" ให้คุณ "แนะนำกลยุทธ์" (recommended_strategy) ที่ดีที่สุดในการแบ่งส่วนนั้นๆ:
    - "semantic": ถ้าส่วนนั้นเป็น "ความเรียง" หรือ "บทความ" ที่ต้องแบ่งตามความหมาย
    - "structural": ถ้าส่วนนั้นมีโครงสร้างชัดเจน (เช่น กฎหมาย, ถาม-ตอบ, ข้อบังคับ)
    - "recursive": ถ้าส่วนนั้นเป็นเนื้อหาทั่วไป หรือคุณไม่แน่ใจ (เป็นค่า Default ที่ปลอดภัย)

[ข้อมูลเอกสาร]
- หัวข้อ: "{document_title}"
- สรุปย่อ: "{summary}"

[เนื้อหาเอกสารทั้งฉบับ (ย่อ)]
{document_preview}

จงตอบกลับเป็น JSON object ที่มี "แผนผังโครงสร้าง" (layout_map) เท่านั้น:

ตัวอย่างผลลัพธ์:
{{
  "layout_map": {{
    "sections": [
      {{
        "section_id": 1,
        "title": "บทนำและหลักการ",
        "char_start": 0,
        "char_end": 4850,
        "recommended_strategy": "semantic"
      }},
      {{
        "section_id": 2,
        "title": "ระเบียบข้อบังคับ มาตรา 1-50",
        "char_start": 4851,
        "char_end": 28900,
        "recommended_strategy": "structural"
      }},
      {{
        "section_id": 3,
        "title": "ภาคผนวก: คำถามที่พบบ่อย",
        "char_start": 28901,
        "char_end": 35000,
        "recommended_strategy": "recursive" 
      }}
    ]
  }}
}}

**จงส่งคืนเฉพาะ JSON object ที่สมบูรณ์เท่านั้น โดยไม่มีคำอธิบายอื่นใดเพิ่มเติม**
"""
)

def _parse_json_from_llm(text: str) -> dict | None:
    """Helper function to safely parse JSON from LLM response."""
    try:
        match = text[text.find('{'):text.rfind('}')+1]
        return json.loads(match)
    except Exception:
        return None

# ==============================================================================
# สถานีที่ 1: Preprocess Node (ไม่มีการแก้ไข)
# ==============================================================================
def preprocess_node(state: GraphState) -> GraphState:
    print("--- ⚙️ สถานี: Preprocessing ---")
    file_path = state.get("file_path")
    try:
        response = requests.post(f"{API_BASE_URL}/tools/preprocess_document", json={"file_path": file_path})
        response.raise_for_status()
        data = response.json()
        if data.get("status") == "success":
            print("   -> ✅ สกัดและพิสูจน์อักษรสำเร็จ")
            state['clean_text'] = data.get("clean_text")
            state['original_filename'] = os.path.basename(file_path)
        else:
            print(f"   -> ❌ API Error: {data.get('message')}")
            state['error_message'] = data.get('message')
    except requests.exceptions.RequestException as e:
        print(f"   -> ❌ Network Error: {e}")
        state['error_message'] = str(e)
    return state

# ==============================================================================
# สถานีที่ 2: Metadata Node (ไม่มีการแก้ไข)
# ==============================================================================
def metadata_node(state: GraphState) -> GraphState:
    print("--- ⚙️ สถานี: Metadata Generation ---")
    if state.get("error_message"): return state
    try:
        response = requests.post(
            f"{API_BASE_URL}/tools/generate_metadata",
            json={"clean_text": state.get("clean_text"), "original_filename": state.get("original_filename")}
        )
        response.raise_for_status()
        data = response.json()
        print("   -> ✅ สร้าง Metadata สำเร็จ")
        state['metadata'] = data.get("metadata")
    except requests.exceptions.RequestException as e:
        print(f"   -> ❌ Network Error: {e}")
        state['error_message'] = str(e)
    return state

# ==============================================================================
# [V2] สถานีที่ 3: Layout Analysis (แทนที่ Strategize)
# ==============================================================================
def layout_analysis_node(state: GraphState) -> GraphState:
    print("--- 🤔🗺️ สถานี: Layout Analysis (V2 - นักวิเคราะห์โครงสร้าง) ---")
    if state.get("error_message"): return state

    llm = get_llm()
    metadata = state.get("metadata", {})
    clean_text = state.get("clean_text", "")

    # ใช้เนื้อหาตัวอย่าง (เช่น 20000 ตัวอักษร) เพื่อประหยัด Token แต่ก็มากพอ
    preview = clean_text[:20000]

    prompt = LAYOUT_ANALYSIS_PROMPT_V2.format(
        document_title=metadata.get("document_title", ""),
        summary=metadata.get("summary", ""),
        document_text=clean_text, # (ส่งเนื้อหาเต็มให้ LLM วิเคราะห์)
        document_preview=preview  # (ส่งตัวอย่างให้ LLM ดูใน Prompt)
    )

    try:
        print("   -> 🧐 กำลังส่งเนื้อหาให้ LLM ช่วยวิเคราะห์โครงสร้าง...")
        response_text = llm.complete(prompt).text
        layout_data = _parse_json_from_llm(response_text)

        if layout_data and "layout_map" in layout_data:
            print(f"   -> ✅ วิเคราะห์โครงสร้างสำเร็จ พบ {len(layout_data['layout_map'].get('sections', []))} ส่วน")
            state['layout_map'] = layout_data['layout_map']
        else:
            print("   -> ⚠️ วิเคราะห์โครงสร้างล้มเหลว, จะใช้ 'กลยุทธ์เดียว' (Recursive) ทั้งไฟล์")
            # สร้าง Layout Map พื้นฐาน (Fallback)
            fallback_map = {
                "sections": [{
                    "section_id": 1, "title": "Full Document",
                    "char_start": 0, "char_end": len(clean_text),
                    "recommended_strategy": "recursive"
                }]
            }
            state['layout_map'] = fallback_map

    except Exception as e:
        print(f"   -> ❌ เกิดข้อผิดพลาดในการวิเคราะห์โครงสร้าง: {e}")
        state['error_message'] = str(e)

    return state


# ==============================================================================
# [V2] สถานีที่ 4: Chunker (ทำตาม "แผนผัง" และ "คำสั่งแก้")
# ==============================================================================
def chunker_node(state: GraphState) -> GraphState:
    print("--- ⚙️ สถานี: Chunking (V2 - ทำตามแผนผัง) ---")
    if state.get("error_message"): return state

    layout_map = state.get("layout_map")
    
    # --- [V5] ตรวจสอบว่ามี "คำสั่งแก้" จาก Validator หรือไม่ ---
    retry_instructions = {}
    history_list = state.get("retry_history", [])
    if history_list:
        # ดึง "ยา" ล่าสุดที่ "แพทย์" สั่งมา
        last_prescription = history_list[-1].get("prescription_given", {})
        if last_prescription.get("action") == "RETRY_SECTION":
            retry_instructions = last_prescription
            print(f"   -> 💡 ได้รับคำสั่งแก้ (V5) จาก Validator: {retry_instructions}")

    payload = {
        "clean_text": state.get("clean_text"),
        "metadata": state.get("metadata"),
        "original_filename": state.get("original_filename"),
        "layout_map": layout_map, # <-- [V2] ส่ง "แผนผัง" ไปให้เครื่องมือ
        "retry_instructions": retry_instructions # <-- [V5] ส่ง "คำสั่งแก้" ไปให้เครื่องมือ
    }

    try:
        response = requests.post(f"{API_BASE_URL}/tools/create_chunks", json=payload)
        response.raise_for_status()
        data = response.json()
        print(f"   -> ✅ แบ่งเอกสารสำเร็จ ได้ {len(data.get('chunks', []))} Chunks")
        state['chunks'] = data.get("chunks")
    
    except requests.exceptions.RequestException as e:
        print(f"   -> ❌ Network Error: {e}")
        state['error_message'] = str(e)

    return state

# ==============================================================================
# [V5] สถานีที่ 5: Validate Chunks (แพทย์ผู้เชี่ยวชาญ V5)
# ==============================================================================
def validate_chunks_node(state: GraphState) -> GraphState:
    print("--- 🤔🧐🧠 สถานี: Validate Chunks (V5 - แพทย์ผู้เชี่ยวชาญ) ---")
    if state.get("error_message"): return state

    chunks = state.get("chunks", [])
    if not chunks:
        state['error_message'] = "Chunking process returned no chunks."
        return state

    llm = get_llm()
    previous_chunk_text = "ไม่มี"

    # [V5] โหลด "แฟ้มประวัติ"
    history_list = state.get("retry_history", [])
    retry_history_str = "ไม่มี"
    if history_list:
        retry_history_str = json.dumps(history_list, indent=2, ensure_ascii=False)

    # --- เริ่มการตรวจสอบทีละ Chunk ---
    for i, chunk in enumerate(chunks):
        current_chunk_text = chunk.get("content", "")
        if not current_chunk_text: continue

        # [V5] ดึงข้อมูล Metadata ของ Chunk เพื่อบอก "แพทย์" ว่า Chunk นี้มาจากไหน
        chunk_metadata = chunk.get("metadata", {})
        section_id = chunk_metadata.get("section_id", "N/A")
        section_title = chunk_metadata.get("section_title", "N/A")
        strategy_used = chunk_metadata.get("strategy_used", "N/A")

        print(f"   -> 🧐 กำลังตรวจสอบ Chunk #{i+1} (จาก Section: '{section_title}')...")

        # [V5] ใช้ Prompt V5, ส่ง "แฟ้มประวัติ" และ "ข้อมูล Section"
        prompt = ULTIMATE_VALIDATION_PROMPT_V5.format(
            document_title=state.get("metadata", {}).get("document_title", "N/A"),
            previous_chunk_text=previous_chunk_text,
            current_chunk_text=current_chunk_text,
            section_id=section_id,
            section_title=section_title,
            strategy_used=strategy_used,
            retry_history_str=retry_history_str
        )
        
        response_text = llm.complete(prompt).text
        validation_result = _parse_json_from_llm(response_text)

        # [V5] ตรรกะการตัดสินใจ (กรณีไม่ผ่าน)
        if not validation_result or not validation_result.get("is_valid"):
            reason = validation_result.get("reason", "Unknown")
            diagnose = validation_result.get("diagnose", "No diagnosis")
            print(f"   -> ❌ Validation Failed: Chunk #{i+1} คุณภาพไม่ผ่าน.")
            print(f"      -> เหตุผล: {reason}")
            print(f"      -> วินิจฉัย: {diagnose}")

            recommendation = validation_result.get("recommendation")
            
            # [V5] บันทึก "แฟ้มประวัติ"
            if recommendation:
                 full_diagnosis_entry = {
                    "attempt": len(history_list) + 1,
                    "diagnosis": {"reason": reason, "diagnose": diagnose},
                    "prescription_given": recommendation 
                }
                 history_list.append(full_diagnosis_entry)
            else:
                recommendation = {"action": "GIVE_UP"}
                history_list.append({"attempt": len(history_list) + 1, "diagnosis": "Malformed LLM response", "prescription_given": recommendation})
            
            state['retry_history'] = history_list

            action = recommendation.get("action")
            
            if action == "GIVE_UP":
                print("   -> 💡 วินิจฉัย (LLM): ยอมแพ้ (GIVE_UP).")
                state['error_message'] = f"Validation failed (LLM recommendation: GIVE_UP)"
                state['validation_passes'] = 0 
                return state

            elif action == "RETRY_SECTION":
                print(f"   -> 💡 วินิจฉัย (LLM): สั่งลองใหม่ที่ Section ID: {recommendation.get('target_section_id')}")
                # ไม่ต้องทำอะไรเพิ่ม "เภสัชกร" (chunker_node) จะอ่าน "ยา" (RETRY_SECTION)
                # จาก `retry_history` เองในรอบถัดไป
                pass

            state['validation_passes'] = 0 
            return state # <-- ออกจาก Node ทันทีเพื่อวนกลับไปทำใหม่

        previous_chunk_text = current_chunk_text

    # --- [V5] ถ้า 'for loop' จบ (ผ่านทุก Chunks) ---
    print("   -> ✅ Validation Passed: คุณภาพ Chunks ทั้งหมดอยู่ในเกณฑ์ดีเยี่ยม")
    state['validation_passes'] = 1
    state['retry_history'] = [] 
    return state

# ==============================================================================
# สถานีที่ 5: Indexer Node (ไม่มีการแก้ไข)
# ==============================================================================
def index_node(state: GraphState) -> GraphState:
    print("--- ⚙️ สถานี: Indexing ---")
    if state.get("error_message"): return state
    try:
        response = requests.post(
            f"{API_BASE_URL}/tools/index_document",
            json={
                "clean_text": state.get("clean_text"),
                "metadata": state.get("metadata"),
                "chunks": state.get("chunks"),
                "original_filename": state.get("original_filename")
            }
        )
        response.raise_for_status()
        data = response.json()
        if data.get("success"):
            print("   -> ✅ บันทึกข้อมูลลงฐานข้อมูลสำเร็จ!")
        else:
            state['error_message'] = data.get("message")
            print(f"   -> ❌ Indexing Failed: {data.get('message')}")
    except requests.exceptions.RequestException as e:
        print(f"   -> ❌ Network Error: {e}")
        state['error_message'] = str(e)
    return state