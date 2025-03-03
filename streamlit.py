import streamlit as st
import subprocess

def main():
    """Streamlit UI for interacting with the RAG-based system."""
    st.title("AI-Powered Research Assistant")
    st.write("Enter your query below and get AI-generated answers based on retrieved context from research papers.")
    
    query = st.text_input("Enter your query:", "")
    
    if st.button("Generate Answer"):
        if query.strip():
            # Run the main script with the provided query
            result = subprocess.run(["python", "main.py"], capture_output=True, text=True)
            
            st.subheader("Response:")
            st.write(result.stdout)
        else:
            st.warning("Please enter a query before generating an answer.")

if __name__ == "__main__":
    main()
