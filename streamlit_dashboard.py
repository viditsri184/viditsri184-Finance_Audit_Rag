# streamlit_dashboard.py
import streamlit as st
import requests
import pandas as pd
import os

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Finance Audit RAG Dashboard")
st.title("Finance Audit RAG — Query Dashboard")

if "history" not in st.session_state:
    st.session_state.history = []

query = st.text_input("Ask about SOX controls, filings, etc.")
if st.button("Send"):
    if query.strip():
        resp = requests.post(f"{API_URL}/query", json={"query": query})
        data = resp.json()
        entry = {"query": query, "answer": data.get("answer", ""), "source": data.get("source", ""), "time": __import__("datetime").datetime.utcnow()}
        st.session_state.history.insert(0, entry)

st.write("Query history")
if st.session_state.history:
    df = pd.DataFrame(st.session_state.history)
    st.dataframe(df)
else:
    st.info("No queries yet — try asking about SOX controls.")
