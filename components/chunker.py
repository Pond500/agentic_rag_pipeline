# agentic_rag_pipeline/components/chunker.py (เวอร์ชัน V2 + V5)

import re
from typing import List, Dict, Any, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.schema import Document

from agentic_rag_pipeline.core.llm_provider import get_embed_model 


# --- [V5+V2] เพิ่ม Import ที่ขาดไป ---
from llama_index.embeddings.huggingface import HuggingFaceEmbedding # <-- บรรทัดนี้จะใช้ได้แล้ว
from agentic_rag_pipeline import config


# --- [V2] อัปเกรด Helper Function 1: Recursive ---
def _recursive_strategy(
    text_piece: str, 
    base_metadata: dict, 
    start_chunk_num: int,  # <-- [V2] รับเลขเริ่มต้น
    chunk_size: int = 1000,
    chunk_overlap: int = 150
) -> List[Dict[str, Any]]:
    """
    กลยุทธ์การแบ่งตามขนาดที่ยืดหยุ่นที่สุด (RecursiveCharacterTextSplitter)
    """
    print(f" -> ใช้กลยุทธ์ Recursive Splitting (Size: {chunk_size}, Overlap: {chunk_overlap})...")
    
    separators = ["\\n\\n", "\\n", " ", ""]
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators
    )
    
    split_texts = text_splitter.split_text(text_piece)
    
    chunks = []
    doc_title = base_metadata.get("document_title", "ไม่ระบุหัวข้อ")
    section_title = base_metadata.get("section_title", "N/A") # <-- [V2] ดึงชื่อ Section

    for i, text in enumerate(split_texts):
        chunk_metadata = base_metadata.copy()
        chunk_metadata["chunk_number"] = start_chunk_num + i # <-- [V2] นับเลขต่อ
        
        # [V2] เพิ่ม Context ของ Section เข้าไป
        enriched_content = f"จากเอกสาร: {doc_title}\nส่วน: {section_title}\n\n{text}"
        
        chunks.append({
            "content": enriched_content,
            "metadata": chunk_metadata
        })
        
    return chunks

# --- [V2] อัปเกรด Helper Function 2: Structural ---
def _structural_strategy(
    text_piece: str, 
    base_metadata: dict,
    start_chunk_num: int  # <-- [V2] รับเลขเริ่มต้น
) -> List[Dict[str, Any]]:
    """
    กลยุทธ์การแบ่งตามโครงสร้างที่ชัดเจน เช่น 'มาตรา', 'บทที่', 'คำถาม:'
    """
    print(" -> พยายามใช้กลยุทธ์ Structural Splitting...")
    
    patterns = [
        r'(\\nมาตรา\\s+\\d+)',     # กฎหมาย
        r'(\\nบทที่\\s+\\d+)',      # ระเบียบ/คู่มือ
        r'(\\nคำถาม:)',          # เอกสาร Q&A
    ]
    
    final_chunks = []
    
    for pattern in patterns:
        if re.search(pattern, text_piece):
            print(f" -> ตรวจพบโครงสร้าง! กำลังแบ่งตาม pattern: {pattern}")
            
            structural_parts = re.split(pattern, text_piece)
            
            combined_parts = []
            if structural_parts[0] and structural_parts[0].strip():
                 combined_parts.append(structural_parts[0].strip())

            for i in range(1, len(structural_parts), 2):
                combined_chunk = (structural_parts[i] + structural_parts[i+1]).strip()
                combined_parts.append(combined_chunk)
            
            # [V2] หลังจากแบ่งตามโครงสร้างแล้ว ให้ใช้ Recursive เพื่อแบ่งชิ้นส่วนที่ยังใหญ่อยู่
            # โดยส่งต่อ global_chunk_counter
            global_chunk_counter = start_chunk_num
            for part in combined_parts:
                sub_chunks = _recursive_strategy(
                    text_piece=part,
                    base_metadata=base_metadata,
                    start_chunk_num=global_chunk_counter, # <-- [V2] ส่งเลขปัจจุบัน
                    chunk_size=1000, # (ใช้ค่า Default สำหรับ sub-chunking)
                    chunk_overlap=150
                )
                final_chunks.extend(sub_chunks)
                global_chunk_counter += len(sub_chunks) # <-- [V2] อัปเดตตัวนับ
            
            return final_chunks 

    print(" -> ไม่พบโครงสร้างที่ชัดเจน")
    return []


# --- [V2] อัปเกรด Helper Function 3: Semantic ---
def _semantic_strategy(
    text_piece: str,
    base_metadata: dict,
    start_chunk_num: int,
    breakpoint_threshold: int = 95 
) -> List[Dict[str, Any]]:
    print(f" -> ใช้กลยุทธ์ Semantic Splitting (Threshold: {breakpoint_threshold})...")
    try:
        # [ใหม่!] สร้าง Embedding Wrapper ของ LlamaIndex โดยตรง
        wrapped_embed_model = HuggingFaceEmbedding(model_name=config.EMBED_MODEL_NAME)

        splitter = SemanticSplitterNodeParser(
            embed_model=wrapped_embed_model, # <--- ส่ง Wrapper ของ LlamaIndex เข้าไปโดยตรง
            breakpoint_percentile_threshold=breakpoint_threshold
        )
        nodes = splitter.get_nodes_from_documents([Document(text=text_piece)])

        chunks = []
        doc_title = base_metadata.get("document_title", "ไม่ระบุหัวข้อ")
        section_title = base_metadata.get("section_title", "N/A") # <-- [V2] ดึงชื่อ Section

        for i, node in enumerate(nodes):
            chunk_metadata = base_metadata.copy()
            chunk_metadata["chunk_number"] = start_chunk_num + i # <-- [V2] นับเลขต่อ
            
            # [V2] เพิ่ม Context ของ Section เข้าไป
            enriched_content = f"จากเอกสาร: {doc_title}\nส่วน: {section_title}\n\n{node.get_content()}"
            chunks.append({"content": enriched_content, "metadata": chunk_metadata})

        return chunks
    except Exception as e:
        print(f"   -> ❌ Semantic Splitting ล้มเหลว: {e}")
        return [] # ถ้าล้มเหลว ให้คืนค่าลิสต์ว่าง

# --- [V2+V5] Main Function (เวอร์ชันอัปเกรด) ---
def create_chunks_for_text(
    text: str,
    metadata: Dict[str, Any],
    original_filename: str,
    layout_map: Dict[str, Any],         # <-- [V2] รับ "แผนผัง"
    retry_instructions: Dict[str, Any]  # <-- [V5] รับ "คำสั่งแก้"
) -> List[Dict[str, Any]]:
    
    print(f"สถานีที่ 3: Agent Chunker (V2) กำลังทำงานตาม 'แผนผัง'...")
    
    all_chunks = []
    global_chunk_counter = 1
    
    sections = layout_map.get("sections", [])
    
    # --- [V2] Fallback กรณีไม่มี "แผนผัง" (Layout Map) ---
    if not sections:
        print("   -> ⚠️ ไม่พบ 'แผนผัง', จะใช้ Recursive กับทั้งเอกสาร")
        sections = [{
            "section_id": 1, 
            "title": "Full Document", 
            "char_start": 0, 
            "char_end": len(text),
            "recommended_strategy": "recursive"
        }]

    # --- [V2] วน Loop ตาม "แผนผัง" (Layout Map) ---
    for section in sections:
        section_id = section.get("section_id")
        title = section.get("title", "N/A")
        start = section.get("char_start", 0)
        end = section.get("char_end", len(text))
        strategy = section.get("recommended_strategy", "recursive")

        # --- [V5] ตรวจสอบว่ามี "คำสั่งแก้" จาก Validator สำหรับ Section นี้หรือไม่ ---
        if retry_instructions and retry_instructions.get("target_section_id") == section_id:
            new_strategy = retry_instructions.get("suggestion", strategy)
            print(f"   -> 💡 V5: ได้รับคำสั่งแก้สำหรับ Section '{title}', เปลี่ยนกลยุทธ์จาก '{strategy}' เป็น '{new_strategy}'")
            strategy = new_strategy # <-- ใช้กลยุทธ์ใหม่ตามที่ "แพทย์" สั่ง

        print(f"   -> ⚙️ กำลังประมวลผล Section: '{title}' (กลยุทธ์: {strategy.upper()})")
        
        # ดึงเนื้อหาเฉพาะส่วน (Section Text)
        section_text = text[start:end]
        if not section_text.strip():
            print(f"   -> ⚠️ ข้าม Section '{title}' เนื่องจากไม่มีเนื้อหา")
            continue
            
        # สร้าง Metadata เฉพาะสำหรับ Chunks ใน Section นี้
        section_metadata = metadata.copy()
        section_metadata["section_id"] = section_id
        section_metadata["section_title"] = title
        section_metadata["strategy_used"] = strategy

        section_chunks = []
        
        # --- [V2] เลือกเครื่องมือตามกลยุทธ์ที่กำหนด ---
        if strategy == "structural":
            section_chunks = _structural_strategy(section_text, section_metadata, global_chunk_counter)
            # [V2] Fallback
            if not section_chunks:
                print("   -> ⚠️ Structural ล้มเหลว, ใช้ Recursive เป็นแผนสำรอง")
                section_chunks = _recursive_strategy(section_text, section_metadata, global_chunk_counter)
        
        elif strategy == "semantic":
            section_chunks = _semantic_strategy(section_text, section_metadata, global_chunk_counter)
            # [V2] Fallback
            if not section_chunks:
                print("   -> ⚠️ Semantic ล้มเหลว, ใช้ Recursive เป็นแผนสำรอง")
                section_chunks = _recursive_strategy(section_text, section_metadata, global_chunk_counter)
        
        else: # Default to "recursive"
            section_chunks = _recursive_strategy(section_text, section_metadata, global_chunk_counter)
            
        all_chunks.extend(section_chunks)
        global_chunk_counter += len(section_chunks) # <-- [V2] อัปเดตตัวนับสำหรับ Section ถัดไป

    print(f"   -> ✅ สร้าง Chunks ทั้งหมด {len(all_chunks)} ชิ้น จาก {len(sections)} ส่วน")
    return all_chunks