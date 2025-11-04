# app.py

import streamlit as st
from utils import process_contract

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="AI Auto Contract Review Bot",
    page_icon="🤖",
    layout="wide"
)

# --- Main Application UI ---
def main():
    st.title("AI-Powered Auto Contract Review Bot 🤖")
    st.markdown("""
    Welcome! This tool uses Google's Gemini AI to help you analyze legal contracts.
    
    **How it works:**
    1.  **Upload your contract** (PDF or DOCX format).
    2.  Click the **"Review Contract"** button.
    3.  The AI will generate a summary, highlight potentially risky clauses, and suggest edits.
    
    **Disclaimer:** This is an AI-powered tool and not a substitute for professional legal advice. Always consult with a qualified lawyer for critical matters.
    """)

    # --- File Uploader ---
    uploaded_file = st.file_uploader(
        "Choose your contract file",
        type=['pdf', 'docx'],
        help="Please upload your contract in .pdf or .docx format."
    )

    if uploaded_file is not None:
        st.success(f"File '{uploaded_file.name}' uploaded successfully!")

        # --- Review Button ---
        if st.button("Review Contract", type="primary"):
            with st.spinner("Analyzing your contract... This may take a moment."):
                try:
                    # Process the contract using the utility function
                    analysis_result = process_contract(uploaded_file)

                    # Display the results
                    st.subheader("📝 Contract Analysis Report")
                    st.markdown(analysis_result)

                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    st.error("Please ensure your GOOGLE_API_KEY is correctly set in the .env file and that the file format is supported.")

# --- Application Entry Point ---
if __name__ == '__main__':
    main()