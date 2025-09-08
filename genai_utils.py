import os
import streamlit as st
from dotenv import load_dotenv
import pandas as pd
from langchain_groq import ChatGroq

def setup(key: str):
    if "STREAMLIT_RUNTIME" in os.environ:
        try:
            return st.secrets[key]
        except:
            st.error("GROQ_API_KEY not found in api.env")
            st.stop()
    else:
        try:
            load_dotenv("api.env")
            groq_api_key = os.getenv(key)
            return groq_api_key
        except:
            st.error("GROQ_API_KEY not found in api.env")
            st.stop()
            
def get_llm(api_key):
    return ChatGroq(
        api_key=api_key,
        model_name="openai/gpt-oss-20b",
        temperature=0.3
    )
