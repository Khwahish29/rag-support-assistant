import streamlit as st
from backend.rag_pipeline import ask_support_assistant

st.title("🛠 RAG-Powered Support Assistant")

query = st.text_input("Enter your issue:")
if st.button("Get Solution"):
    response = ask_support_assistant(query)
    st.write("### Suggested Resolution")
    st.write(response)
