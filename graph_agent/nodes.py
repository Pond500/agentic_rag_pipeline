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
ULTIMATE_VALIDATION_PROMPT = PromptTemplate.from_template(
    """คุณคือ "สุดยอดบรรณาธิการ AI" (Master Editor AI) ของสำนักพิมพ์ดิจิทัล ภารกิจของคุณคือการตรวจสอบคุณภาพของ "Chunk ปัจจุบัน" อย่างเข้มงวดที่สุด โดยพิจารณาจาก 3 มิติหลัก เพื่อให้แน่ใจว่าเนื้อหาทั้งหมดจะถูกเรียบเรียงออกมาอย่างสมบูรณ์แบบที่สุด

---
### **ข้อมูลประกอบการพิจารณา:**

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
### **ภารกิจการตรวจสอบ (Your Mission):**

จงวิเคราะห์ข้อมูลข้างต้นอย่างละเอียด และประเมินคุณภาพของ **"Chunk ปัจจุบัน"** ตามเกณฑ์ต่อไปนี้ แล้วตอบกลับเป็น JSON object ที่สมบูรณ์เท่านั้น:

1.  **Integrity (ความสมบูรณ์ในตัวเอง):**
    - "Chunk ปัจจุบัน" เป็นประโยคที่สมบูรณ์, อ่านรู้เรื่อง, และไม่ถูกตัดจบกลางคันใช่หรือไม่?

2.  **Cohesion (ความต่อเนื่องทางบริบท):**
    - "Chunk ปัจจุบัน" มีเนื้อหาที่ต่อเนื่องและสมเหตุสมผลกับ "Chunk ก่อนหน้า" หรือไม่? มันเป็นเรื่องราวที่ตามกันมา หรือเป็นการขึ้นหัวข้อใหม่ที่สมเหตุสมผล?

3.  **Relevance (ความเกี่ยวข้องกับหัวข้อหลัก):**
    - เนื้อหาใน "Chunk ปัจจุบัน" ยังคงเกี่ยวข้องกับ "หัวข้อหลักของเอกสาร" หรือไม่ หรือเป็นเนื้อหาที่ดูเหมือนจะหลุดประเด็นไปอย่างชัดเจน?

**จงสรุปผลการประเมินของคุณลงใน JSON object ที่มีโครงสร้างดังนี้เท่านั้น:**

{{
  "is_valid": boolean,
  "reason": "อธิบายเหตุผลของการตัดสินใจของคุณอย่างชัดเจนและกระชับ หากไม่ผ่าน ให้ระบุว่าไม่ผ่านเพราะเกณฑ์ข้อไหน (Integrity, Cohesion, or Relevance) และเพราะอะไร (เช่น 'incomplete sentence', 'context shift', 'irrelevant topic')"
}}
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
# สถานีที่ 4: Validate Chunks Node (เวอร์ชัน "สุดยอดบรรณาธิการ")
# ==============================================================================
def validate_chunks_node(state: GraphState) -> GraphState:
    print("--- 🤔🧐🧠 สถานี: Validate Chunks (ใช้สมอง สุดยอดบรรณาธิการ) ---")
    if state.get("error_message"): return state

    chunks = state.get("chunks", [])
    metadata = state.get("metadata", {})
    document_title = metadata.get("document_title", "ไม่ระบุ")

    if not chunks:
        state['error_message'] = "Chunking process returned no chunks."
        return state

    llm = get_llm()
    previous_chunk_text = "ไม่มี"

    for i, chunk in enumerate(chunks):
        current_chunk_text = chunk.get("content", "")
        if not current_chunk_text: continue

        print(f"   -> 🧐 กำลังตรวจสอบคุณภาพ Chunk #{i+1} (เทียบกับ Chunk ก่อนหน้า)...")

        prompt = ULTIMATE_VALIDATION_PROMPT.format(
            document_title=document_title,
            previous_chunk_text=previous_chunk_text[:1000], # ใช้แค่บางส่วนเพื่อประหยัด Token
            current_chunk_text=current_chunk_text[:1000]
        )
        response_text = llm.complete(prompt).text
        validation_result = _parse_json_from_llm(response_text)

        if not validation_result or not validation_result.get("is_valid"):
            reason = validation_result.get("reason", "Unknown") if validation_result else "Malformed LLM response"
            print(f"   -> ❌ Validation Failed: Chunk #{i+1} คุณภาพไม่ผ่าน. เหตุผล: {reason}")

            # --- ส่วนการวินิจฉัยและสั่งยา (สามารถทำให้ซับซ้อนขึ้นได้อีกในอนาคต) ---
            new_params = {}
            if "cohesion" in reason.lower() or "context" in reason.lower():
                print("   -> 💡 วินิจฉัย: บริบทอาจขาดหาย ลองเพิ่ม Overlap")
                new_params["chunk_overlap"] = 250 # ลองเพิ่ม Overlap
            else:
                print("   -> 💡 วินิจฉัย: ปัญหาอาจอยู่ที่ขนาด ลองลดขนาด Chunk")
                new_params["chunk_size"] = 700

            state['chunking_params'] = new_params
            state['chunking_retries'] = state.get('chunking_retries', 0) + 1
            state['validation_passes'] = 0
            return state

        # อัปเดต previous_chunk_text สำหรับการวนลูปครั้งถัดไป
        previous_chunk_text = current_chunk_text

    print("   -> ✅ Validation Passed: คุณภาพ Chunks ทั้งหมดอยู่ในเกณฑ์ดีเยี่ยม")
    state['validation_passes'] = 1
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