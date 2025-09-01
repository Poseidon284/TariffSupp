import streamlit as st
import atexit
import tempfile
import os
import base64
import sqlite3
import streamlit.components.v1 as components

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from RAG_utils import rag_chain, upsert_file_to_chroma, del_collection

def cleanup():
    del_collection()

atexit.register(cleanup)

# ---------------------------
# Streamlit page setup
# ---------------------------
st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("📄 RAG Chatbot with Supplier and Tariff Documents")

# ---------------------------
# SESSION STATE INIT
# ---------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "pdf_path" not in st.session_state:
    st.session_state["pdf_path"] = None
if "doc_type" not in st.session_state:
    st.session_state["doc_type"] = None
if "pdf_processed" not in st.session_state:
    st.session_state["pdf_processed"] = False

# ---------------------------
# SIDEBAR → PDF Upload + Viewer
# ---------------------------
with st.sidebar:
    st.header("📑 PDF Panel")

    doc_type = st.text_input(
        "Document type & name", 
        placeholder="e.g. Tariff-UAE2024", 
        value=st.session_state["doc_type"] or ""
    )
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"], key="pdf_uploader")

    if uploaded_file and not st.session_state["pdf_processed"]:
        # Save uploaded file temporarily
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Store in session state
        st.session_state["pdf_path"] = temp_path
        st.session_state["doc_type"] = doc_type

        # Ingest into Chroma (only once per upload)
        with st.spinner("📥 Processing and indexing PDF..."):
            upsert_file_to_chroma(temp_path, doc_type=doc_type)

        st.session_state["pdf_processed"] = True

    # Show PDF if already uploaded
    if st.session_state["pdf_path"]:
        with open(st.session_state["pdf_path"], "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode("utf-8")
        iframe = f"""
                <iframe src="{st.session_state["pdf_path"]}" 
                        width="100%" height="700px" 
                        style="border:none; border-radius:8px;" 
                        allowfullscreen></iframe>
                """
        # st.markdown(
        #     f"""
        #     <iframe src="data:application/pdf;base64,{base64_pdf}" 
        #             width="100%" height="700px" 
        #             style="border:none; border-radius:8px;" 
        #             allowfullscreen></iframe>
        #     """,
        #     unsafe_allow_html=False
        # )
        components.html(iframe, width="100%", height="700px")
    else:
        st.info("👆 Upload a PDF to view it here")

# ---------------------------
# MAIN AREA → Chatbot
# ---------------------------
# Display chat history
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg:
            st.markdown("**Sources:**")
            for s in msg["sources"]:
                st.markdown(f"- [{s['doc_type']}] {s['source']}: {s['text']}")

# Chat input
if user_input := st.chat_input("Type your question..."):
    # Store user message
    st.session_state["messages"].append({"role": "user", "content": user_input})

    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get response from RAG
    result = rag_chain(user_input)

    bot_msg = {
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"]
    }
    st.session_state["messages"].append(bot_msg)

    # Display assistant reply immediately
    with st.chat_message("assistant"):
        st.markdown(result["answer"])
        # if result["sources"]:
        #     st.markdown("**Sources:**")
        #     for s in result["sources"]:
        #         st.markdown(f"- [{s['doc_type']}] {s['source']}: {s['text']}")
