# agentic_rag_pipeline/inspector_app.py (เวอร์ชัน Control Room)

from graph_agent.graph import graph_app
import os
import streamlit as st
import pandas as pd
import psycopg2
import tempfile
import pprint

# --- Import ส่วนประกอบจากโปรเจกต์ Agent ของเรา ---
# ตอนนี้การ Import นี้จะทำงานได้แล้ว
import config
from graph_agent.graph import graph_app

# --- Database Connection Function (เหมือนเดิม) ---
@st.cache_resource
def get_db_connection():
    try:
        conn = psycopg2.connect(
            dbname=config.DB_NAME, user=config.DB_USER, password=config.DB_PASS,
            host=config.DB_HOST, port=config.DB_PORT
        )
        return conn
    except psycopg2.OperationalError as e:
        st.error(f"การเชื่อมต่อฐานข้อมูลล้มเหลว: {e}")
        return None

# --- Main App ---
st.set_page_config(layout="wide", page_title="Agent Pipeline Inspector")
st.title("🔬 Agentic Pipeline Inspector & Control Room")

conn = get_db_connection()
if not conn:
    st.error("ไม่สามารถเชื่อมต่อฐานข้อมูลได้ กรุณาตรวจสอบ .env และสถานะของ Docker")
    st.stop()

# --- สร้าง Tabs ---
tab_control, tab_bots, tab_kb, tab_chunks = st.tabs([
    "🕹️ Agent Control Room", "🤖 Bot & Persona Inspector",
    "📚 Knowledge Base Inspector", "🧩 Chunk Viewer"
])

# ==============================================================================
# TAB 1: AGENT CONTROL ROOM (ส่วนใหม่ที่ทรงพลังที่สุด!)
# ==============================================================================
with tab_control:
    st.header("🕹️ สั่งการและติดตามการทำงานของ LangGraph Agent")
    st.markdown("อัปโหลดไฟล์เอกสารเพื่อเริ่ม Pipeline และดูสถานะการทำงานของ Agent ในแต่ละขั้นตอนแบบ Real-time")

    uploaded_file = st.file_uploader(
        "เลือกไฟล์เอกสาร (.pdf, .docx, .txt) ที่ต้องการประมวลผล",
        type=['pdf', 'docx', 'txt']
    )

    if uploaded_file is not None:
        # สร้าง temporary file เพื่อให้ Agent มี path ที่แน่นอนสำหรับอ่านไฟล์
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            file_path = tmp_file.name
        
        st.info(f"ไฟล์พร้อมสำหรับประมวลผลที่: `{file_path}`")

        if st.button("🚀 เริ่มการทำงานของ Agent", use_container_width=True, type="primary"):
            st.markdown("---")
            st.subheader("🔴 LIVE: ติดตามการทำงานของ Agent")
            status_container = st.container()

            # --- เตรียม "ถาด" (State) ใบแรก ---
            initial_state = {
                "file_path": file_path,
                "original_filename": os.path.basename(file_path), # <-- เพิ่ม OS.PATH
                "clean_text": "",
                "metadata": {},
                "chunks": [],
                "error_message": None,
                "layout_map": {},        # <-- [V2] Field ใหม่ที่จำเป็น
                "validation_passes": 0,
                "retry_history": []      # <-- [V5] Field ใหม่ที่จำเป็น (แทนที่ chunking_retries)
            }

            try:
                
                for step in graph_app.stream(initial_state):
                    # `step` คือ Dictionary ที่มี key เป็นชื่อ Node ที่เพิ่งทำงานเสร็จ
                    node_name = list(step.keys())[0]
                    node_state = step[node_name]

                    with status_container:
                        with st.expander(f"**สถานี: `{node_name}`** - ทำงานเสร็จสิ้น", expanded=True):
                            
                            # --- [V5+V2] Smart Display Logic ---
                            if node_name == "layout_analysis":
                                st.markdown("##### 🗺️ แผนผังโครงสร้าง (Layout Map)")
                                st.json(node_state.get("layout_map", {}))
                            
                            elif node_name == "validate_chunks":
                                st.markdown("##### 🩺 แฟ้มประวัติการรักษา (Retry History)")
                                st.json(node_state.get("retry_history", []))
                                if node_state.get("validation_passes", 0) > 0:
                                    st.success("-> ✅ คุณภาพผ่าน!")
                                if node_state.get("error_message"):
                                    st.error(f"-> 🛑 ยอมแพ้: {node_state.get('error_message')}")
                            
                            elif node_name == "chunker":
                                st.markdown(f"##### 🧩 ได้รับ Chunks ทั้งหมด: {len(node_state.get('chunks', []))} ชิ้น")
                            
                            else:
                                # ถ้าเป็น Node อื่นๆ ให้แสดงผลแบบเดิม
                                st.code(pprint.pformat(node_state), language="json")

                st.success("🎉 Pipeline ทำงานเสร็จสิ้นสมบูรณ์!")

            except Exception as e:
                st.error(f"เกิดข้อผิดพลาดร้ายแรงระหว่างการทำงานของ Agent: {e}")
            finally:
                # ลบ temporary file ทิ้งเมื่อทำงานเสร็จ
                if os.path.exists(file_path):
                    os.remove(file_path)


# ==============================================================================
# TAB 2: BOT & PERSONA INSPECTOR (เหมือนเดิม)
# ==============================================================================
with tab_bots:
    st.header("🤖 ตรวจสอบภาพรวม Bots และ Personas")
    st.markdown("ดูภาพรวมของบอททั้งหมดในระบบ และบุคลิกภาพ (Prompts) ที่แต่ละตัวใช้งาน")

    try:
        # Query ข้อมูล Bot และ Persona ทั้งหมดโดยการ JOIN สองตาราง
        query = """
            SELECT
                b.id as bot_id,
                b.bot_name,
                b.api_key,
                b.qdrant_collection_name,
                b.is_active,
                p.id as persona_id,
                p.persona_name,
                p.system_prompt,
                p.routing_prompt,
                p.refusal_message,
                p.about_bot_message
            FROM bots b
            LEFT JOIN bot_personas p ON b.persona_id = p.id
            ORDER BY b.id;
        """
        bots_df = pd.read_sql(query, conn)

        st.success(f"พบ **{len(bots_df)}** บอทในระบบ")

        # แสดงตารางภาพรวมของบอท
        st.dataframe(
            bots_df[['bot_id', 'bot_name', 'api_key', 'qdrant_collection_name', 'is_active', 'persona_name']],
            use_container_width=True,
            hide_index=True
        )

        st.divider()
        st.subheader("🔍 ดูรายละเอียด Persona (Prompts)")

        # สร้าง Expander สำหรับแต่ละ Persona ที่พบ
        for index, row in bots_df.iterrows():
            with st.expander(f"**{row['bot_name']}** (Persona: '{row['persona_name']}')"):
                st.markdown(f"**Bot ID:** `{row['bot_id']}` | **API Key:** `{row['api_key']}`")
                st.text_area("System Prompt (บุคลิกหลัก)", row['system_prompt'], height=200, disabled=True, key=f"sys_{index}")
                st.text_area("Routing Prompt (ตัวจัดประเภท)", row['routing_prompt'], height=200, disabled=True, key=f"route_{index}")
                st.text_area("Refusal Message (ข้อความปฏิเสธ)", row['refusal_message'], height=100, disabled=True, key=f"refuse_{index}")
                st.text_area("About Bot Message (ข้อความแนะนำตัว)", row['about_bot_message'], height=100, disabled=True, key=f"about_{index}")

    except Exception as e:
        st.error(f"ไม่สามารถโหลดข้อมูล Bots และ Personas ได้: {e}")



# ==============================================================================
# TAB 3 & 4: KNOWLEDGE BASE & CHUNK VIEWER (เหมือนเดิม)
# ==============================================================================
with tab_kb:
    st.header("📚 ภาพรวมเอกสารใน `knowledge_items`")
    try:
        items_df = pd.read_sql("SELECT id, title, source_type, status, created_at FROM knowledge_items ORDER BY id DESC", conn)
        st.dataframe(items_df, use_container_width=True, hide_index=True)
    except Exception as e:
        st.error(f"ไม่สามารถโหลดข้อมูล knowledge_items ได้: {e}")


with tab_chunks:
    st.header("🧩 ตรวจสอบหน่วยข้อมูลย่อย (Chunk Viewer)")
    item_id_to_view = st.number_input(
        "ใส่ ID ของเอกสารที่ต้องการดู Chunks (ดู ID จากตาราง 'Knowledge Base Inspector')",
        min_value=1,
        step=1
    )
    if st.button("🔬 แสดง Chunks", use_container_width=True):
        if item_id_to_view > 0:
            try:
                chunks_df = pd.read_sql(
                    "SELECT chunk_sequence, chunk_text, metadata FROM knowledge_chunks WHERE knowledge_item_id = %s ORDER BY chunk_sequence",
                    conn,
                    params=(item_id_to_view,)
                )
                st.info(f"พบ **{len(chunks_df)}** Chunks สำหรับเอกสาร ID: {item_id_to_view}")

                for index, row in chunks_df.iterrows():
                    chunk_length = len(row['chunk_text'])
                    expander_title = f"Chunk #{row['chunk_sequence']} (ความยาว: {chunk_length} ตัวอักษร)"
                    with st.expander(expander_title):
                        st.text_area("Content", row['chunk_text'], height=200, disabled=True, key=f"chunk_detail_{index}")
                        st.json(row['metadata'])

            except Exception as e:
                st.error(f"ไม่สามารถโหลด Chunks ได้: {e}")