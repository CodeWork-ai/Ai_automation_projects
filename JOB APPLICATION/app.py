import streamlit as st
from dotenv import load_dotenv
from modules.text_extractor import extract_text
from modules.groq_helper import (
    generate_response,
    tailor_resume_prompt,
    draft_cover_letter_prompt,
    generate_interview_questions_prompt
)

# Load environment variables from .env file
load_dotenv()

# --- Streamlit App Configuration ---
st.set_page_config(page_title="AI Career Assistant", layout="wide")

st.title("🚀 AI-Powered Career Assistant")
st.markdown("Upload your resume and paste a job description to get started!")

# --- Session State Initialization ---
if 'resume_text' not in st.session_state:
    st.session_state.resume_text = ""

# --- Layout and User Inputs ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("📄 Your Resume")
    uploaded_resume = st.file_uploader("Upload your resume (PDF or DOCX)", type=["pdf", "docx"])
    if uploaded_resume:
        with st.spinner("Extracting text from resume..."):
            st.session_state.resume_text = extract_text(uploaded_resume)
        if st.session_state.resume_text:
            st.success("Resume text extracted successfully!")
            with st.expander("View Extracted Resume Text"):
                # UPDATED with label and label_visibility to remove warning
                st.text_area(
                    "Extracted Resume Text",
                    st.session_state.resume_text,
                    height=200,
                    label_visibility="collapsed"
                )
        else:
            st.error("Could not extract text from the resume. Please try a different file.")

with col2:
    st.subheader("💼 Job Description")
    jd_text = st.text_area("Paste the job description here", height=300)

# --- Feature Buttons and Logic ---
st.header("✨ Choose Your Action")

action = st.radio(
    "Select what you want to do:",
    ("Tailor Resume", "Draft Cover Letter", "Generate Interview Questions"),
    horizontal=True
)

if st.button(f"Generate {action}"):
    # Ensure inputs are not empty before proceeding
    is_ready = True
    if action in ["Tailor Resume", "Draft Cover Letter"]:
        if not st.session_state.resume_text or not jd_text:
            st.warning("Please upload a resume and paste a job description.")
            is_ready = False
    elif action == "Generate Interview Questions":
        if not jd_text:
            st.warning("Please paste a job description.")
            is_ready = False

    if is_ready:
        if action == "Tailor Resume":
            prompt = tailor_resume_prompt(st.session_state.resume_text, jd_text)
            with st.spinner("Generating resume tailoring suggestions..."):
                response = generate_response(prompt)
                st.subheader("Resume Tailoring Suggestions")
                st.markdown(response)

        elif action == "Draft Cover Letter":
            prompt = draft_cover_letter_prompt(st.session_state.resume_text, jd_text)
            with st.spinner("Drafting your cover letter..."):
                response = generate_response(prompt)
                st.subheader("Drafted Cover Letter")
                st.markdown(response)

        elif action == "Generate Interview Questions":
            prompt = generate_interview_questions_prompt(jd_text)
            with st.spinner("Generating interview questions..."):
                response = generate_response(prompt)
                st.subheader("Potential Interview Questions")
                st.markdown(response)