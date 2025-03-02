import streamlit as st
import time

def generate_response(user_input):
    # Simulate AI-generated response
    responses = {
        "hello": "Hi there! How can I assist you?",
        "who are you": "I am a simple AI chatbot built with Streamlit!",
        "what is AI": "AI stands for Artificial Intelligence, which enables machines to mimic human cognition.",
        "bye": "Goodbye! Have a great day!"
    }
    return responses.get(user_input.lower(), "I'm not sure about that, but I'm learning every day!")

# Streamlit UI
st.title("🤖 Simple Streamlit Chatbot")
st.write("Ask me anything!")

# User input
user_input = st.text_input("Your Question:")

if st.button("Get Response"):
    with st.spinner("Thinking..."):
        time.sleep(1)
        response = generate_response(user_input)
        st.success(response)

# Run this app using `streamlit run streamlit.py`