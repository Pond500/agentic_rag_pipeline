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
ULTIMATE_VALIDATION_PROMPT_V4 = PromptTemplate.from_template(
    """คุณคือ "แพทย์ผู้เชี่ยวชาญด้านการแบ่งข้อมูล AI" (AI Chunking Strategist & Doctor) ภารกิจของคุณคือการตรวจสอบคุณภาพของ "Chunk ปัจจุบัน" อย่างเข้มงวดที่สุด และสั่งการแก้ไขหากจำเป็น

---
### **ข้อมูลคนไข้ (Chunk ปัจจุบัน):**

1.  **หัวข้อหลักของเอกสาร (Main Topic):** "{document_title}"

2.  **Chunk ก่อนหน้า (Previous Chunk):**
    (หากเป็น Chunk แรกสุด ส่วนนี้จะระบุว่า "ไม่มี")
    ---
    {previous_chunk_text}
    ---

3.  **Chunk ปัจจุบันที่ต้องตรวจสอบ (Current Chunk):**
    ---
    {current_chunk_text}
    ---

---
### **แฟ้มประวัติการรักษา (ที่ล้มเหลว):**
(นี่คือสิ่งที่เคยลองทำไปแล้ว และผลลัพธ์คือ "ไม่ผ่าน")
---
{retry_history_str}
---

### **พารามิเตอร์ปัจจุบันที่ใช้ (ที่เพิ่งล้มเหลว):**
(นี่คือการตั้งค่าที่ทำให้เกิด Chunk ปัจจุบัน)
---
{current_params_str}
---

---
### **ภารกิจการตรวจสอบ (Your Mission):**

จงวิเคราะห์ข้อมูลทั้งหมด และประเมินคุณภาพของ **"Chunk ปัจจุบัน"**

**หากคุณภาพผ่าน (is_valid: true):**
ให้ตอบกลับ JSON ที่มี "is_valid" เป็น true เท่านั้น

**หากคุณภาพไม่ผ่าน (is_valid: false):**
1.  ให้อธิบาย "เหตุผล" (reason) และ "วินิจฉัย" (diagnose)
2.  ให้อ่าน "แฟ้มประวัติการรักษา" และ "พารามิเตอร์ปัจจุบัน"
3.  ให้ "สั่งยา" (recommendation) สูตรใหม่ที่ **ฉลาด** และ **ไม่ซ้ำเดิม** เพื่อแก้ปัญหา
4.  **[สำคัญ]** ถ้าวิเคราะห์ "แฟ้มประวัติ" แล้วพบว่าลองมาหลายทาง (เช่น ลองทั้งปรับพารามิเตอร์ และ ลองทั้งเปลี่ยนกลยุทธ์) แล้วยังล้มเหลว ให้สั่ง `action: "GIVE_UP"` เพื่อยอมแพ้

**จงตอบกลับเป็น JSON object ที่มีโครงสร้างดังนี้เท่านั้น:**
{{
  "is_valid": boolean,
  "reason": "อธิบายเหตุผลที่ 'ไม่ผ่าน' (เช่น 'ประโยคถูกตัดจบกลางคัน')",
  "diagnose": "วิเคราะห์สาเหตุ (เช่น 'chunk_size เล็กเกินไปสำหรับย่อหน้านี้')",
  "recommendation": {{
    "action": "เลือกหนึ่งอย่าง: ADJUST_PARAMS | CHANGE_STRATEGY | GIVE_UP",
    "strategy": "(ถ้า action=CHANGE_STRATEGY) ระบุ 'semantic', 'structural', 'recursive'",
    "new_chunk_size": "(ถ้า action=ADJUST_PARAMS) ระบุ 'ตัวเลขใหม่' (เช่น 1350) หรือ null",
    "new_chunk_overlap": "(ถ้า action=ADJUST_PARAMS) ระบุ 'ตัวเลขใหม่' (เช่น 250) หรือ null"
  }}
}}

**ตัวอย่างผลลัพธ์ (กรณีสั่งยา):**
{{
  "is_valid": false,
  "reason": "บริบทยังขาดหาย แม้จะเพิ่ม overlap แล้ว",
  "diagnose": "การเพิ่ม overlap ที่ 200 (จากพารามิเตอร์ปัจจุบัน) ยังไม่พอ",
  "recommendation": {{
    "action": "ADJUST_PARAMS",
    "strategy": null,
    "new_chunk_size": 1200,
    "new_chunk_overlap": 300
  }}
}}

**ตัวอย่างผลลัพธ์ (กรณียอมแพ้):**
{{
  "is_valid": false,
  "reason": "ลองทั้งปรับ size และเปลี่ยนเป็น semantic แล้ว แต่เนื้อหาต้นฉบับ (OCR) เละเกินไป",
  "diagnose": "เอกสารต้นทางมีปัญหา (Unfixable)",
  "recommendation": {{
    "action": "GIVE_UP",
    "strategy": null,
    "new_chunk_size": null,
    "new_chunk_overlap": null
  }}
}}

**จงส่งคืนเฉพาะ JSON object ที่สมบูรณ์เท่านั้น โดยไม่มีคำอธิบายอื่นใดเพิ่มเติม**
"""
)

STRATEGY_PROMPT = PromptTemplate.from_template(
    """คุณคือ "นักวางกลยุทธ์การแบ่งข้อมูล" (Chunking Strategist)
    หน้าที่ของคุณคือวิเคราะห์ข้อมูลสรุปของเอกสาร แล้วเลือกกลยุทธ์การแบ่ง (Chunking Strategy) ที่ "เหมาะสมที่สุด" เพียง 1 อย่างจากรายการต่อไปนี้:

    [ตัวเลือกกลยุทธ์]
    1. "structural": เหมาะสำหรับเอกสารที่มีโครงสร้างชัดเจนและคาดเดาได้ เช่น กฎหมาย (มาตรา), ระเบียบ (ข้อ), หรือบทสนทนา (คำถาม-คำตอบ)
    2. "semantic": เหมาะสำหรับเอกสารที่เป็นความเรียง, บทความ, หรือคู่มือ ที่ต้องการรักษา "กลุ่มก้อนของความหมาย" และตัดแบ่งเมื่อมีการเปลี่ยนเรื่องอย่างชัดเจน
    3. "recursive": เป็นวิธีมาตรฐานที่ปลอดภัยที่สุด เหมาะสำหรับเอกสารทั่วไปที่ไม่มีโครงสร้างชัดเจน หรือเมื่อคุณไม่แน่ใจ

    [ข้อมูลสรุปของเอกสาร]
    - หัวข้อ: "{document_title}"
    - สรุปย่อ: "{summary}"
    - ตัวอย่างเนื้อหา: "{content_preview}"

    จงวิเคราะห์ข้อมูลสรุป แล้วตอบกลับด้วย "ชื่อของกลยุทธ์ที่ดีที่สุด" เพียงคำเดียวเท่านั้น (เช่น "structural", "semantic", "recursive")
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
# [ใหม่!] สถานีที่ 3: Strategize Chunking Node (นักวางกลยุทธ์)
# ==============================================================================
def strategize_chunking_node(state: GraphState) -> GraphState:
    """
    Node นี้คือ "นักวางกลยุทธ์" ที่ใช้ LLM ตัดสินใจเลือก Chunking Strategy
    """
    print("--- 🤔♟️ สถานี: Strategist (วางกลยุทธ์การแบ่งข้อมูล) ---")
    if state.get("error_message"): return state

    llm = get_llm()
    metadata = state.get("metadata", {})
    clean_text = state.get("clean_text", "")

    prompt = STRATEGY_PROMPT.format(
        document_title=metadata.get("document_title", ""),
        summary=metadata.get("summary", ""),
        content_preview=clean_text[:500] # ใช้ 500 ตัวอักษรแรกเป็นตัวอย่าง
    )

    try:
        print("   -> 🧐 กำลังส่งข้อมูลให้ LLM ช่วยเลือกกลยุทธ์...")
        response = llm.complete(prompt).text
        # ทำความสะอาดคำตอบของ LLM ให้เหลือแค่ชื่อกลยุทธ์
        strategy = response.strip().lower().replace('"', '').replace("'", "")

        if strategy not in ["structural", "semantic", "recursive"]:
            print(f"   -> ⚠️ LLM ตอบกลับไม่ถูกต้อง ('{strategy}'), ใช้ 'recursive' เป็นค่าเริ่มต้น")
            strategy = "recursive"

        print(f"   -> ✅ LLM ตัดสินใจเลือกกลยุทธ์: '{strategy}'")
        state['chunking_strategy'] = strategy

    except Exception as e:
        print(f"   -> ❌ เกิดข้อผิดพลาดในการเลือกกลยุทธ์, ใช้ 'recursive' เป็นค่าเริ่มต้น. Error: {e}")
        state['chunking_strategy'] = "recursive"

    return state


# ==============================================================================
# สถานีที่ 4: Chunker Node (เวอร์ชันปรับปรุง: ทำตามคำสั่ง)
# ==============================================================================
def chunker_node(state: GraphState) -> GraphState:
    print("--- ⚙️ สถานี: Chunking (ทำตามคำสั่ง) ---")
    if state.get("error_message"): return state

    params = state.get("chunking_params", {})
    # [ใหม่!] ดึงกลยุทธ์ที่ "นักวางกลยุทธ์" เลือกไว้จาก State
    strategy = state.get("chunking_strategy", "recursive")

    print(f"   -> ทำงานตามกลยุทธ์: '{strategy}' พร้อมคำแนะนำ: {params}")

    payload = {
        "clean_text": state.get("clean_text"),
        "metadata": state.get("metadata"),
        "original_filename": state.get("original_filename"),
        "strategy": strategy, # <-- ส่งกลยุทธ์ที่ถูกเลือก
        **params
    }

    try:
        response = requests.post(f"{API_BASE_URL}/tools/create_chunks", json=payload)
        response.raise_for_status()
        data = response.json()
        print(f"   -> ✅ แบ่งเอกสารสำเร็จ ได้ {len(data.get('chunks', []))} Chunks")
        state['chunks'] = data.get("chunks")
        state['chunking_params'] = {}
    except requests.exceptions.RequestException as e:
        print(f"   -> ❌ Network Error: {e}")
        state['error_message'] = str(e)

    return state

# ==============================================================================
# สถานีที่ 4: Validate Chunks Node (เวอร์ชัน V4 - แพทย์ผู้เชี่ยวชาญ + แฟ้มประวัติ)
# ==============================================================================
def validate_chunks_node(state: GraphState) -> GraphState:
    print("--- 🤔🧐🧠 สถานี: Validate Chunks (V4 - แพทย์ผู้เชี่ยวชาญ + แฟ้มประวัติ) ---")
    if state.get("error_message"): 
        return state

    chunks = state.get("chunks", [])
    metadata = state.get("metadata", {})
    document_title = metadata.get("document_title", "ไม่ระบุ")

    if not chunks:
        state['error_message'] = "Chunking process returned no chunks."
        return state

    llm = get_llm()
    previous_chunk_text = "ไม่มี"

    # --- [V4] โหลด "ประวัติการรักษา" (แฟ้มประวัติ) ---
    history_list = state.get("retry_history", [])
    retry_history_str = "ไม่มี"
    if history_list:
        try:
            retry_history_str = json.dumps(history_list, indent=2, ensure_ascii=False)
        except Exception:
            retry_history_str = str(history_list)

    # --- [V4] โหลด "พารามิเตอร์ปัจจุบัน" ที่เพิ่งใช้ ---
    current_params = state.get("chunking_params", {})
    # (ตั้งค่าเริ่มต้น หากยังไม่มี)
    if not current_params:
         current_params = {"chunk_size": 1000, "chunk_overlap": 150} # (ค่าเริ่มต้นที่สมมติขึ้น)
    
    current_params_str = json.dumps(current_params, indent=2, ensure_ascii=False)


    # --- เริ่มการตรวจสอบทีละ Chunk ---
    for i, chunk in enumerate(chunks):
        current_chunk_text = chunk.get("content", "")
        if not current_chunk_text: 
            continue

        print(f"   -> 🧐 กำลังตรวจสอบคุณภาพ Chunk #{i+1} (เทียบกับ Chunk ก่อนหน้า)...")

        # --- [V4] ใช้ Prompt V4, ส่ง "แฟ้มประวัติ" และ "พารามิเตอร์ปัจจุบัน" ---
        prompt = ULTIMATE_VALIDATION_PROMPT_V4.format(
            document_title=document_title,
            previous_chunk_text=previous_chunk_text,   # (ไม่จำกัด Token)
            current_chunk_text=current_chunk_text,     # (ไม่จำกัด Token)
            retry_history_str=retry_history_str,       # <-- ส่งแฟ้มประวัติ
            current_params_str=current_params_str      # <-- ส่งพารามิเตอร์ที่เพิ่งใช้
        )
        
        response_text = llm.complete(prompt).text
        validation_result = _parse_json_from_llm(response_text)

        # --- [V4] ตรรกะการตัดสินใจ (กรณีไม่ผ่าน) ---
        if not validation_result or not validation_result.get("is_valid"):
            reason = validation_result.get("reason", "Unknown") if validation_result else "Malformed LLM response"
            diagnose = validation_result.get("diagnose", "No diagnosis") if validation_result else "No diagnosis"
            print(f"   -> ❌ Validation Failed: Chunk #{i+1} คุณภาพไม่ผ่าน.")
            print(f"      -> เหตุผล: {reason}")
            print(f"      -> วินิจฉัย: {diagnose}")

            recommendation = validation_result.get("recommendation")
            
            # --- [V4] บันทึก "แฟ้มประวัติ" (อาการ + ยาที่สั่ง) ---
            if recommendation:
                 full_diagnosis_entry = {
                    "attempt": len(history_list) + 1,
                    "diagnosis": {
                        "reason": reason,
                        "diagnose": diagnose
                    },
                    "prescription_given": recommendation # "ยา" ที่แพทย์สั่งในรอบนี้
                }
                 history_list.append(full_diagnosis_entry)
            else:
                # กรณีฉุกเฉิน: LLM ตอบ JSON ไม่ครบ
                recommendation = {"action": "GIVE_UP"} # สั่งให้ยอมแพ้ถ้า LLM งง
                history_list.append({"attempt": len(history_list) + 1, "diagnosis": "Malformed LLM response", "prescription_given": recommendation})
            
            state['retry_history'] = history_list # <-- อัปเดต State

            action = recommendation.get("action")
            
            # --- [V4] เภสัชกรจ่ายยาตามที่แพทย์ (LLM) สั่ง ---

            if action == "GIVE_UP":
                print("   -> 💡 วินิจฉัย (LLM): ยอมแพ้ (GIVE_UP). ปัญหานี้อาจแก้ไขไม่ได้")
                # [สำคัญ] ส่งสัญญาณ Error ให้ "ทางแยก" (should_continue) รู้
                state['error_message'] = f"Validation failed (LLM recommendation: GIVE_UP after {len(history_list)} attempts)"
                state['validation_passes'] = 0 
                return state # <-- ออกจาก Node (ทางแยก จะจับ error_message และสั่ง END)

            elif action == "CHANGE_STRATEGY":
                new_strategy = recommendation.get("strategy", "recursive")
                print(f"   -> 💡 วินิจฉัย (LLM): ต้องเปลี่ยนกลยุทธ์เป็น '{new_strategy}'")
                state['chunking_strategy'] = new_strategy # <-- สั่งเปลี่ยนกลยุทธ์
                state['chunking_params'] = {} # <-- รีเซ็ตพารามิเตอร์
            
            elif action == "ADJUST_PARAMS":
                print(f"   -> 💡 วินิจฉัย (LLM): ต้องปรับพารามิเตอร์...")
                
                # ใช้ค่าปัจจุบันเป็นฐาน
                new_params = current_params.copy()

                # [V4] อ่าน "ใบสั่งยา" (ตัวเลขใหม่) จาก LLM
                new_size = recommendation.get("new_chunk_size")
                new_overlap = recommendation.get("new_chunk_overlap")

                if new_size is not None:
                    new_params["chunk_size"] = int(new_size)
                    print(f"      -> สั่งยา (LLM): เปลี่ยน chunk_size เป็น {new_params['chunk_size']}")
                    
                if new_overlap is not None:
                    new_params["chunk_overlap"] = int(new_overlap)
                    print(f"      -> สั่งยา (LLM): เปลี่ยน chunk_overlap เป็น {new_params['chunk_overlap']}")

                state['chunking_params'] = new_params # <-- อัปเดต State ด้วยยาชุดใหม่
            
            else:
                # กรณีฉุกเฉิน: LLM สั่ง action ที่ไม่รู้จัก
                print("   -> 💡 วินิจฉัย (ฉุกเฉิน): LLM สั่ง action ที่ไม่รู้จัก, จะลองลด Size")
                new_params = current_params.copy()
                new_params["chunk_size"] = max(300, int(new_params.get("chunk_size", 1000) * 0.8))
                state['chunking_params'] = new_params

            state['validation_passes'] = 0 # <-- ตั้งค่าว่า "ยังไม่ผ่าน"
            return state # <-- ออกจาก Node ทันทีเพื่อวนกลับไปทำใหม่

        # ถ้า Chunk นี้ผ่าน (is_valid: true) ก็ให้อัปเดต previous_chunk_text แล้วไปตรวจชิ้นถัดไป
        previous_chunk_text = current_chunk_text

    # --- [V4] ถ้า 'for loop' จบ (ผ่านทุก Chunks) ---
    print("   -> ✅ Validation Passed: คุณภาพ Chunks ทั้งหมดอยู่ในเกณฑ์ดีเยี่ยม")
    state['validation_passes'] = 1  # <-- ตั้งค่าว่า "ผ่านแล้ว"
    state['retry_history'] = []   # <-- [สำคัญ!] ล้างประวัติการรักษาเมื่อหายดีแล้ว
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