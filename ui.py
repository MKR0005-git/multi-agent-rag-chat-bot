import streamlit as st
from fastapi import FastAPI
import requests

# Streamlit UI Setup
st.set_page_config(page_title="Multi-Agent RAG Chatbot", page_icon="🤖", layout="wide")

# Title and Description
st.title("🤖 Multi-Agent RAG Chatbot")
st.markdown("""
### Ask any question and get responses powered by an advanced Retrieval-Augmented Generation system!
""")

# Input field for user query
user_query = st.text_input("Enter your question:", placeholder="Type your question here...")

# Backend API URL (Modify this to match your FastAPI endpoint)
API_URL = "http://127.0.0.1:8000/query"

# Function to fetch response from FastAPI backend
def get_response(query):
    try:
        response = requests.post(API_URL, json={"query": query})
        return response.json().get("response", "Sorry, I couldn't fetch a response.")
    except Exception as e:
        return f"Error: {str(e)}"

# Button to send query
if st.button("Ask AI"):  
    if user_query.strip():
        with st.spinner("Thinking..."):
            bot_response = get_response(user_query)
            st.write("### 🤖 Response:")
            st.success(bot_response)
    else:
        st.warning("Please enter a question!")

# Footer
st.markdown("---")
st.markdown("**Made with ❤️ using Streamlit & FastAPI**")
