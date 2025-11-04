import streamlit as st
import json
import os
import re
from dotenv import load_dotenv
from phi.assistant import Assistant
from phi.llm.groq import Groq
from fpdf import FPDF
import io

st.set_page_config(
    page_title="AI Course Generator",
    page_icon="🎓",
    layout="wide"
)

load_dotenv()

def load_css():
    css = """
    body, .css-18e3th9 {
        background-color: #0d1117;
        color: #c9d1d9;
        font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
    }
    .stButton>button, .sidebar-outline-btn {
        background-color: #0366d6;
        color: white;
        font-weight: bold;
        width: 100% !important;
        min-width: 180px;
        max-width: 220px;
        min-height: 48px;
        border-radius: 8px;
        margin-bottom: 12px;
        text-align: center;
        transition: background .2s;
        box-sizing: border-box;
    }
    .stButton>button:hover, .sidebar-outline-btn:hover {
        background-color: #044289;
        color: white;
    }
    .section-container {
        background-color: #161b22;
        padding: 1.5rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        box-shadow: 0 0 10px #21262d;
    }
    .sub-topic {
        background-color: #21262d;
        margin-top: 0.8rem;
        margin-bottom: 0.8rem;
        padding: 1rem;
        border-radius: 6px;
        font-size: 14px;
    }
    """
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

load_css()

def create_pdf_bytes(sub_title, sub_explanation, sub_code_example):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, sub_title, ln=True)

    pdf.set_font("Arial", "", 12)
    for line in sub_explanation.split("\n"):
        pdf.multi_cell(0, 7, line)
        pdf.ln(1)

    pdf.set_font("Courier", "", 10)
    pdf.set_text_color(0, 0, 255)
    for line in sub_code_example.split("\n"):
        pdf.multi_cell(0, 6, line)
    pdf.ln(2)

    pdf_data = pdf.output(dest="S").encode("latin-1")
    return io.BytesIO(pdf_data)

def create_full_course_pdf(course_title, sections):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 14, course_title, ln=True)

    for section in sections:
        pdf.set_font("Arial", "B", 16)
        pdf.ln(6)
        pdf.cell(0, 12, section.get("title", ""), ln=True)

        pdf.set_font("Arial", "", 12)
        pdf.multi_cell(0, 8, section.get("introduction", ""))
        pdf.ln(3)

        key_points = section.get("key_points", [])
        if key_points:
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 9, "Key Takeaways:", ln=True)
            pdf.set_font("Arial", "I", 12)
            for p in key_points:
                pdf.multi_cell(0, 7, f"- {p}")
            pdf.ln(2)

        for sub in section.get("sub_topics", []):
            if isinstance(sub, dict):
                pdf.set_font("Arial", "B", 12)
                pdf.cell(0, 8, sub.get("sub_title", ""), ln=True)
                pdf.set_font("Arial", "", 11)
                pdf.multi_cell(0, 7, sub.get("sub_explanation", ""))
                pdf.ln(1)
                code = sub.get("sub_code_example", "")
                if code.strip():
                    pdf.set_font("Courier", "", 9)
                    for line in code.split("\n"):
                        pdf.multi_cell(0, 6, line)
                pdf.ln(2)

    pdf_data = pdf.output(dest="S").encode("latin-1")
    return io.BytesIO(pdf_data)


GROQ_API_KEY = os.getenv("GROQ_API_KEY")

@st.cache_resource
def get_outline_assistant():
    return Assistant(
        llm=Groq(
            model="openai/gpt-oss-20b",
            response_format={"type": "json_object"},
            api_key=GROQ_API_KEY
        ),
        description="You are an expert curriculum planner.",
        instructions=[
            "Produce ONLY a single valid JSON object with a 'table_of_contents' key whose value is a list of simple section titles as strings.",
            "Do not include any additional text."
        ]
    )

@st.cache_resource
def get_section_assistant():
    return Assistant(
        llm=Groq(
            model="llama-3.3-70b-versatile",
            response_format={"type": "json_object"},
            api_key=GROQ_API_KEY
        ),
        description="You are an expert content writer creating detailed lecture material for a given section.",
        instructions=[
            "Given a section title and main topic, produce a detailed JSON object for that section with keys:",
            "'title' (string), 'introduction' (string), 'key_points' (list of strings), and 'sub_topics' (list of dicts).",
            "Each sub_topic must include 'sub_title' (string), 'sub_explanation' (detailed multi-paragraph string), and 'sub_code_example' (string with raw code, no markdown code fences).",
            "Return ONLY this JSON object with no extra output."
        ]
    )

def extract_json(response_str):
    match = re.search(r"\{.*\}", response_str, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    else:
        raise ValueError("No valid JSON object found in response")

@st.cache_data(show_spinner=False)
def generate_outline(topic: str) -> list:
    assistant = get_outline_assistant()
    prompt = f"Create a table of contents (list of section titles) for a beginner programming course on '{topic}'."
    try:
        response_str = assistant.run(prompt, stream=False)
        json_obj = extract_json(response_str)
        return json_obj.get("table_of_contents", [])
    except Exception as e:
        if "rate limit" in str(e).lower() or "429" in str(e):
            st.warning("Rate limit exceeded during outline generation. Please wait and try again later.")
            return []
        else:
            st.error(f"Error during outline generation: {e}")
            return []

@st.cache_data(show_spinner=False)
def generate_section(topic: str, section_title: str) -> dict:
    assistant = get_section_assistant()
    prompt = f"Create detailed content for the section titled '{section_title}' in a beginner {topic} programming course."
    try:
        response_str = assistant.run(prompt, stream=False)
        section_content = extract_json(response_str)
        return section_content
    except Exception as e:
        if "rate limit" in str(e).lower() or "429" in str(e):
            st.warning(f"Rate limit exceeded while generating section '{section_title}'. Please wait and try again later.")
            return {}
        else:
            st.error(f"Error during section generation: {e}")
            return {}

st.title("🎓 AI-Powered Course Generator")
st.markdown("Generate an in-depth programming course.")

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found! Please add it in your .env file.")
else:
    if "content" not in st.session_state:
        st.session_state.content = {}
    if "topic" not in st.session_state:
        st.session_state.topic = ""
    if "selected_section_title" not in st.session_state:
        st.session_state.selected_section_title = None

    if "section_contents" not in st.session_state:
        st.session_state.section_contents = {}

    def handle_generate_outline(topic):
        st.session_state.topic = topic
        toc = generate_outline(topic)
        st.session_state.content["table_of_contents"] = toc
        st.session_state.section_contents = {}
        if toc:
            st.session_state.selected_section_title = toc[0]

    def handle_generate_section(topic, section_title):
        if section_title not in st.session_state.section_contents:
            content = generate_section(topic, section_title)
            st.session_state.section_contents[section_title] = content

    cols = st.columns(5)
    course_buttons = [
        ("🚀 Generate Python Course", "Python"),
        ("💾 Generate SQL Course", "SQL"),
        ("🌐 Generate HTML Course", "HTML"),
        ("🎨 Generate CSS Course", "CSS"),
        ("💻 Generate JS Course", "JavaScript")
    ]
    for col, (label, topic) in zip(cols, course_buttons):
        if col.button(label, use_container_width=True):
            handle_generate_outline(topic)

    st.markdown("---")

    if st.session_state.content.get("table_of_contents"):
        toc = st.session_state.content["table_of_contents"]

        st.sidebar.title("📖 Course Outline")
        for idx, section_title in enumerate(toc):
            if st.sidebar.button(section_title, key=f"section-{idx}"):
                st.session_state.selected_section_title = section_title
                handle_generate_section(st.session_state.topic, section_title)

        st.header(f"{st.session_state.topic} Programming Course")

        selected_section = st.session_state.selected_section_title
        content = st.session_state.section_contents.get(selected_section)

        if content:
            # Download entire selected section as PDF button
            pdf_bytes = create_full_course_pdf(
                f"{st.session_state.topic} - {content.get('title', '')}",
                [content]
            )
            st.download_button(
                label="📥 Download This Entire Section as PDF",
                data=pdf_bytes,
                file_name=f"{st.session_state.topic}_{selected_section}_FullSection.pdf",
                mime="application/pdf"
            )

            st.markdown(f"<div class='section-container'><h2 style='color:#58a6ff'>{content.get('title', '')}</h2></div>", unsafe_allow_html=True)
            st.markdown(content.get("introduction", ""))
            key_points = content.get("key_points", [])
            if key_points:
                st.markdown("**Key Takeaways:**")
                for point in key_points:
                    if isinstance(point, str):
                        st.markdown(f"- {point}")

            sub_topics = content.get("sub_topics", [])
            for idx, sub in enumerate(sub_topics):
                if isinstance(sub, dict):
                    st.markdown(f"<div class='sub-topic'><h4 style='color:#79c0ff'>{sub.get('sub_title', '')}</h4></div>", unsafe_allow_html=True)
                    st.markdown(sub.get("sub_explanation", ""))
                    code = sub.get("sub_code_example", "")
                    if code.strip():
                        st.code(code, language=st.session_state.topic.lower(), line_numbers=True)

        else:
            st.info("Select a section from the outline to begin learning.")
