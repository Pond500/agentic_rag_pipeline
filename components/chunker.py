# agentic_rag_pipeline/components/chunker.py

import re
from typing import List, Dict, Any, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.schema import Document

from agentic_rag_pipeline.core.llm_provider import get_embed_model 

def _recursive_strategy(
    text_piece: str, 
    base_metadata: dict, 
    start_chunk_num: int,
    chunk_size: int,      # <--- รับค่าเข้ามา
    chunk_overlap: int  # <--- รับค่าเข้ามา
) -> List[Dict[str, Any]]:
    """
    กลยุทธ์การแบ่งตามขนาดที่ยืดหยุ่นที่สุด (มาจาก RecursiveCharacterTextSplitter)
    """
    print(f" -> ใช้กลยุทธ์ Recursive Splitting...")
    
    # กำหนดตัวแบ่งที่เหมาะสมสำหรับเอกสารทั่วไป
    separators = ["\\n\\n", "\\n", " ", ""]
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators
    )
    
    split_texts = text_splitter.split_text(text_piece)
    
    chunks = []
    doc_title = base_metadata.get("document_title", "ไม่ระบุหัวข้อ")

    for i, text in enumerate(split_texts):
        chunk_metadata = base_metadata.copy()
        chunk_metadata["chunk_number"] = start_chunk_num + i
        
        # เพิ่ม Context เข้าไปในเนื้อหาและ Metadata เพื่อเพิ่มความแม่นยำในการค้นหา
        enriched_content = f"จากเอกสาร: {doc_title}\\n\\n{text}"
        
        chunks.append({
            "content": enriched_content,
            "metadata": chunk_metadata
        })
        
    return chunks

def _structural_strategy(
    full_text: str, 
    base_metadata: dict
) -> List[Dict[str, Any]]:
    """
    กลยุทธ์การแบ่งตามโครงสร้างที่ชัดเจน เช่น 'มาตรา', 'บทที่', 'คำถาม:'
    (ดัดแปลงจาก smart_agent/pipeline/chunker.py)
    """
    print(" -> พยายามใช้กลยุทธ์ Structural Splitting...")
    
    # Regex patterns ที่ใช้ในการค้นหาโครงสร้าง
    patterns = [
        r'(\\nมาตรา\\s+\\d+)',     # กฎหมาย
        r'(\\nบทที่\\s+\\d+)',      # ระเบียบ/คู่มือ
        r'(\\nคำถาม:)',          # เอกสาร Q&A
    ]
    
    final_chunks = []
    
    for pattern in patterns:
        if re.search(pattern, full_text):
            print(f" -> ตรวจพบโครงสร้าง! กำลังแบ่งตาม pattern: {pattern}")
            
            # แบ่งเอกสารตามโครงสร้างที่เจอ
            structural_parts = re.split(pattern, full_text)
            
            # นำ Header ที่ตัดออกมาไปรวมกับเนื้อหาส่วนถัดไป
            combined_parts = []
            if structural_parts[0] and structural_parts[0].strip():
                 combined_parts.append(structural_parts[0].strip())

            for i in range(1, len(structural_parts), 2):
                combined_chunk = (structural_parts[i] + structural_parts[i+1]).strip()
                combined_parts.append(combined_chunk)
            
            # หลังจากแบ่งตามโครงสร้างแล้ว ให้ใช้ Recursive เพื่อแบ่งชิ้นส่วนที่ยังใหญ่อยู่
            global_chunk_counter = 1
            for part in combined_parts:
                sub_chunks = _recursive_strategy(
                    text_piece=part,
                    base_metadata=base_metadata,
                    start_chunk_num=global_chunk_counter
                )
                final_chunks.extend(sub_chunks)
                global_chunk_counter += len(sub_chunks)
            
            return final_chunks # ถ้าเจอโครงสร้างและจัดการแล้ว ให้ออกจากฟังก์ชันเลย

    # ถ้าไม่เจอโครงสร้างที่รู้จักเลย ให้คืนค่าลิสต์ว่าง
    print(" -> ไม่พบโครงสร้างที่ชัดเจน")
    return []


# --- [ใหม่!] Strategy 3: Semantic (Cinematic) ---
def _semantic_strategy(
    full_text: str,
    base_metadata: dict,
    breakpoint_threshold: int = 95 # ค่าความไวในการตัด (0-100)
) -> List[Dict[str, Any]]:
    print(f" -> ใช้กลยุทธ์ Semantic Splitting (Threshold: {breakpoint_threshold})...")
    try:
        embed_model = get_embed_model()

        splitter = SemanticSplitterNodeParser(
            embed_model=embed_model,
            breakpoint_percentile_threshold=breakpoint_threshold
        )
        nodes = splitter.get_nodes_from_documents([Document(text=full_text)])

        chunks = []
        doc_title = base_metadata.get("document_title", "ไม่ระบุหัวข้อ")
        for i, node in enumerate(nodes):
            chunk_metadata = base_metadata.copy()
            chunk_metadata["chunk_number"] = i + 1
            enriched_content = f"จากเอกสาร: {doc_title}\n\n{node.get_content()}"
            chunks.append({"content": enriched_content, "metadata": chunk_metadata})

        return chunks
    except Exception as e:
        print(f"   -> ❌ Semantic Splitting ล้มเหลว: {e}")
        return [] # ถ้าล้มเหลว ให้คืนค่าลิสต์ว่าง

# --- Main Function (เวอร์ชันอัปเกรด) ---
def create_chunks_for_text(
    text: str,
    metadata: Dict[str, Any],
    original_filename: str,
    # --- [ใหม่!] เพิ่มพารามิเตอร์สำหรับ "กลยุทธ์" ---
    strategy: str = "recursive", # <-- ค่าเริ่มต้นคือ recursive
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None
) -> List[Dict[str, Any]]:
    print(f"สถานีที่ 3: Agent Chunker กำลังทำงานด้วยกลยุทธ์: '{strategy.upper()}'")

    # --- [ใหม่!] Logic การเลือกใช้เครื่องมือตามกลยุทธ์ที่ได้รับ ---
    if strategy == "structural":
        chunks = _structural_strategy(text, metadata)
        # ถ้า structural ล้มเหลว ให้ใช้ recursive เป็นแผนสำรอง
        if not chunks:
            print("   -> Structural ล้มเหลว, ใช้ Recursive เป็นแผนสำรอง")
            return _recursive_strategy(text, metadata, 1, 1000, 150)
        return chunks

    elif strategy == "semantic":
        chunks = _semantic_strategy(text, metadata)
        # ถ้า semantic ล้มเหลว ให้ใช้ recursive เป็นแผนสำรอง
        if not chunks:
            print("   -> Semantic ล้มเหลว, ใช้ Recursive เป็นแผนสำรอง")
            return _recursive_strategy(text, metadata, 1, 1000, 150)
        return chunks

    else: # Default to recursive
        final_chunk_size = chunk_size if chunk_size is not None else 1000
        final_chunk_overlap = chunk_overlap if chunk_overlap is not None else 150
        return _recursive_strategy(text, metadata, 1, final_chunk_size, final_chunk_overlap)