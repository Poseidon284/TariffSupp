import os
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain_core.prompts import ChatPromptTemplate
import camelot  # for tables
from PyPDF2 import PdfReader
import pandas as pd
from genai_utils import setup, get_llm
import sqlite3


__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# ---- Clients ----
api_key = setup("GROQ_API_KEY")
groq_llm = get_llm(api_key)

# ---- Get Chromadb Collection ----
def init_chroma():
    persist_directory = "./chroma_data"
    try:
        chroma_client = chromadb.PersistentClient(path=persist_directory)
        collection = chroma_client.get_or_create_collection("Embeddings")
    finally:
        return collection

# ---- Embeddings ----
def get_embeddings(texts, model="all-MiniLM-L6-v2"):
    embedding_function = SentenceTransformerEmbeddingFunction(model_name=model)
    try:
        return [embedding for embedding in embedding_function(texts)]
    except:
        return []

# ---- Text Extraction ----
def extract_text_from_pdf(file_path):
    if file_path.lower().endswith(".pdf"):
        reader = PdfReader(file_path)
        return [page.extract_text() for page in reader.pages if page.extract_text()]
    
    elif file_path.lower().endswith(".csv"):
        df = pd.read_csv(file_path)
        return df.astype(str).values.tolist()  # convert all rows to list of strings
    
    elif file_path.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(file_path)
        return df.astype(str).values.tolist()
    
    else:
        raise ValueError("Unsupported file format. Supported: PDF, CSV, XLSX, XLS")

def extract_tables_from_pdf(file_path):
    try:
        tables = camelot.read_pdf(file_path, pages="all")
        tables.export('camtables.csv', f='csv', compress=True)
        return [t.df for t in tables]
    except:
        return []

# ---- Chunking helper ----
def chunk_text(text, chunk_size=500, overlap=50):
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunks.append(text[start:end])
        if end == text_length:  
            break
        start += chunk_size - overlap 

    return chunks

#For Table Chunking
def chunk_dataframe(df, chunk_size=500, overlap=50):
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    step = chunk_size - overlap
    num_rows = len(df)

    start = 0
    while start < num_rows:
        end = min(start + chunk_size, num_rows)
        chunk = df.iloc[start:end].to_dict(orient="records")
        chunks.append(chunk)
        if end == num_rows:
            break
        start += step

    return chunks

# For Table Embedding
def get_table_chunk_embeddings(table_chunks):
    documents = []
    metadatas = []

    for chunk in table_chunks:
        chunk_text = " | ".join([f"{k}: {v}" for row in chunk for k, v in row.items()])
        documents.append(chunk_text)
        metadatas.append({"rows": str(chunk[0])}) 

    embeddings = get_embeddings(documents)
    return embeddings, documents, metadatas


# ---- Ingest into Chroma ----
def upsert_file_to_chroma(file_path, file_name, doc_type="general"):
    store = init_chroma()
    text_chunks = []
    text = ""
    tables = []
    if doc_type=="application/pdf":
        tables = extract_tables_from_pdf(file_path)
        text = extract_text_from_pdf(file_path)
    elif doc_type=='text/csv':
        tab_df = pd.read_csv(file_path, index_col=None)
        tables.append(tab_df)
        text = chunk_text(tables[0].to_string())
    elif doc_type=='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' or doc_type=='application/vnd.ms-excel':
        tab_df = pd.read_excel(file_path, index_col=None)
        tables.append(tab_df)
        text = chunk_text(tables[0].to_string())
    else:
        return "Incorrect File type"
    table_chunks = []
    for table in tables:
        table_chunks.extend(chunk_dataframe(table, chunk_size=500, overlap=50))

    for t in text:
            text_chunks.extend(chunk_text(t))
    
    if not text_chunks:
        print(f"No extractable text from {file_path}")  
        return
    
    embeddings = get_embeddings(text_chunks)
    metadatas = [{"doc_type": file_name, "source": os.path.basename(file_path)} for _ in text_chunks]
    ids = [f"doc_{i}" for i in range(len(text_chunks))]
    
    store.upsert(
        documents=text_chunks,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )

    embeddings, documents, metadatas = get_table_chunk_embeddings(table_chunks)
    ids = [f'doc_{i}' for i in range(len(documents))]
    
    try:
        store.upsert(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
    except:
        print("Yay no tables!")

    print(f"✅ Uploaded {file_path} as {doc_type}")

# ---- Query Chroma ----
def query_chroma(query, store, n_results=5):
    query_emb = get_embeddings([query])
    results = store.query(query_embeddings=query_emb, n_results=n_results)
    
    chunks = []
    for i in range(len(results["documents"][0])):
        chunks.append({
            "text": results["documents"][0][i],
            "doc_type": results["metadatas"][0][i]["doc_type"],
            "source": results["metadatas"][0][i]["source"]
        })
    return chunks

# ---- RAG Answer with Groq ----
def rag_answer(query, n_results=5):
    store = init_chroma()
    chunks = query_chroma(query, store, n_results=n_results)
    context = "\n\n".join([f"[{c['doc_type'].upper()} | {c['source']}] {c['text']}" for c in chunks])
    
    prompt = ChatPromptTemplate.from_template(
        "You are a helpful assistant answering based on supplier and tariff documents.\n"
        "Query: {query}\n\n"
        "Relevant Context:\n{context}\n\n"
        "Answer the query clearly. You are alloweed to explain who you are to the user if they ask. If unsure about the answer, say you don't know. Do not give answers outside the relevant context. Assign a Risk score based on how sound the clauses are from the questions."
    )
    
    chain = prompt | groq_llm
    response = chain.invoke({"query": query, "context": context})
    
    return {
        "answer": response.content,
        "sources": chunks
    }

def rag_chain(query):
    result = rag_answer(query)
    return result

def del_collection():
    try:
        persist_directory = "./chroma_data"
        chroma_client = chromadb.PersistentClient(path=persist_directory)
        collection = "Embeddings"
        chroma_client.delete_collection(name=collection)

        print("Collection Embeddings deleted successfully.")
    except Exception as e:
        print(f"Error deleting collection: {e}")
    finally:
        return
