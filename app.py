import streamlit as st
import atexit
from streamlit_pdf_viewer import pdf_viewer
from RAG_utils import rag_chain, upsert_file_to_chroma, del_collection
import tempfile
import os
import sqlite3

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

def cleanup():
    del_collection()

atexit.register(cleanup)

# This app clutters your temp files so delete it after running
st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("📄 RAG Chatbot with Supplier and Tariff Documents")

# Layout with two columns: Chatbot (left), PDF Viewer (right)
col1, col2 = st.columns([2, 1])

# ---------------------------
# LEFT COLUMN → Chatbot
# ---------------------------
with col1:
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Display chat history
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sources" in msg:
                st.markdown("**Sources:**")
                for s in msg["sources"]:
                    st.markdown(f"- [{s['doc_type']}] {s['source']}: {s['text']}")

    # Chat input
    user_input = st.chat_input("Type your question...")

    if user_input:
        # Store user message
        st.session_state["messages"].append({"role": "user", "content": user_input})
        
        # Call your RAG chain
        result = rag_chain(user_input)

        # Bot response
        bot_msg = {
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"]
        }
        st.session_state["messages"].append(bot_msg)

        # Display bot response immediately
        with st.chat_message("assistant"):
            st.markdown(result["answer"])
            st.markdown("**Sources:**")
            for s in result["sources"]:
                st.markdown(f"- [{s['doc_type']}] {s['source']}: {s['text']}")

# ---------------------------
# RIGHT COLUMN → PDF Upload + Viewer
# ---------------------------
with col2:
    st.subheader("📑 PDF Upload & Viewer")
    st.text("Choose type and name of document")
    doc_type = st.text_input("Choose type and name of document - Eg. Tariff-UAE2024")

    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_file:
        # Save to a temporary file
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Ingest into Chroma
        with st.spinner("📥 Processing and indexing PDF..."): 
            upsert_file_to_chroma(temp_path, doc_type=doc_type)

        pdf_viewer(f"{temp_path}", width=700, height=1000, zoom_level=1.2) 
