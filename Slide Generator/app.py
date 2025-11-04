import streamlit as st
import os
import time
import fitz  # PyMuPDF
from docx import Document
from PIL import Image
from llm_handler import generate_presentation_structure
from pptx_generator import create_presentation
from dynamic_template_generator import get_template_themes
from preview_generator import generate_theme_preview

st.set_page_config(page_title="AI Slide Generator Pro", page_icon="✨", layout="wide")

# --- Dark Mode CSS (no changes) ---
st.markdown("""
<style>
    /* --- Main App Styling --- */
    .stApp {
        background-color: #1E1E1E;
        color: #EAEAEA;
    }
    
    /* --- Main Title --- */
    h1, h3 {
        color: #FFFFFF;
    }

    /* --- Template Preview Cards (Compact) --- */
    .template-card {
        border: 1px solid #444444;
        border-radius: 8px;
        padding: 0;
        text-align: center;
        transition: transform 0.2s, box-shadow 0.2s;
        background-color: #2D2D2D;
        overflow: hidden;
        height: 100%;
    }
    .template-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 16px rgba(0,0,0,0.3);
        border-color: #7B61FF;
    }
    .template-card img {
        border-bottom: 1px solid #444444;
    }
    .card-content {
        padding: 12px;
    }
    .template-name {
        font-weight: 500;
        color: #FFFFFF;
        font-size: 0.9rem;
        margin-bottom: 10px;
    }

    /* --- Button Styling --- */
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        width: 100%;
    }
    .stButton>button:not([kind="primary"]) {
        background-color: #F0F2F6;
        color: #1E1E1E; /* Dark text for visibility */
        border: 1px solid #555;
    }
    button[kind="primary"], .stDownloadButton>button {
        background-color: #7B61FF;
        height: 3rem;
        border: none;
    }
    button[kind="primary"]:hover, .stDownloadButton>button:hover {
        background-color: #6A50E5;
    }
    
    /* --- Input & Text Area Styling --- */
    .stTextInput>div>div>input, .stTextArea>div>textarea {
        background-color: #333;
        color: #EAEAEA;
        border: 1px solid #555;
    }

    /* --- Tab Styling --- */
    .stTabs [data-baseweb="tab-list"] {
		gap: 12px;
	}
	.stTabs [data-baseweb="tab"] {
		height: 45px;
        background-color: #333333;
        border-radius: 8px;
        padding: 0 20px;
        color: #A0A0A0;
	}
	.stTabs [aria-selected="true"] {
  		background-color: #7B61FF;
        color: white;
        font-weight: 600;
	}
</style>
""", unsafe_allow_html=True)

# --- CACHED FUNCTION for previews (no changes) ---
@st.cache_data
def load_all_previews():
    themes = get_template_themes()
    return {name: generate_theme_preview(details) for name, details in themes.items()}

# --- UTILITY FUNCTIONS for text extraction (no changes) ---
def extract_text_from_pdf(file):
    return "".join(page.get_text() for page in fitz.open(stream=file.read(), filetype="pdf"))

def extract_text_from_docx(file):
    return "\n".join(para.text for para in Document(file).paragraphs)

# --- SESSION STATE INITIALIZATION (no changes) ---
if 'selected_template' not in st.session_state: st.session_state.selected_template = None
if 'pptx_file' not in st.session_state: st.session_state.pptx_file = None

# --- UI RENDERING ---
st.markdown("<h1 style='text-align: center; color: #FFFFFF;'>✨ AI Slide Generator Pro</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #A0A0A0; font-size: 1.1rem; margin-bottom: 2rem;'>Transform your ideas, documents, or topics into polished presentations in seconds.</p>", unsafe_allow_html=True)

# --- 2-COLUMN LAYOUT ---
left_col, right_col = st.columns([2, 3])

# --- LEFT COLUMN: INPUT & ACTIONS ---
with left_col:
    st.subheader("1. Provide Your Content")
    prompt_text = ""
    
    tab1, tab2, tab3, tab4 = st.tabs(["💡 Topic", "✍️ Text", "📄 PDF", "📜 DOCX"])
    with tab1:
        topic_text = st.text_input("Enter the topic for your presentation:", placeholder="e.g., 'The Future of Renewable Energy'")
        if topic_text: prompt_text = topic_text
    with tab2:
        pasted_text = st.text_area("Or, paste the full content:", height=250)
        if pasted_text: prompt_text = pasted_text
    with tab3:
        pdf_file = st.file_uploader("Or, upload a PDF file", type="pdf", label_visibility="collapsed")
        if pdf_file:
            prompt_text = extract_text_from_pdf(pdf_file)
            st.success("PDF processed successfully!")
    with tab4:
        docx_file = st.file_uploader("Or, upload a DOCX file", type="docx", label_visibility="collapsed")
        if docx_file:
            prompt_text = extract_text_from_docx(docx_file)
            st.success("DOCX file processed successfully!")
    
    st.write("---") # Visual separator
    
    st.subheader("3. Generate Your Presentation")
    if st.button("🚀 Generate Now", use_container_width=True, type="primary"):
        if not st.session_state.selected_template:
            st.error("Please select a visual template from the right!")
        elif not prompt_text:
            st.error("Please provide content using one of the methods above!")
        else:
            st.session_state.pptx_file = None
            with st.spinner("Step 1/2: 🧠 AI is crafting your detailed content..."):
                slide_structure = generate_presentation_structure(prompt_text)

            if slide_structure:
                with st.spinner("Step 2/2: 🎨 Assembling your PowerPoint presentation..."):
                    output_path = f"output/presentation_{int(time.time())}.pptx"
                    if not os.path.exists("output"): os.makedirs("output")
                    generated_file = create_presentation(slide_structure, st.session_state.selected_template, output_path)
                if generated_file:
                    st.success("🎉 Your presentation is ready!")
                    st.session_state.pptx_file = generated_file
                else:
                    st.error("💥 Oops! Failed to create the PowerPoint file.")
            else:
                st.error("💥 Oops! The AI did not return a valid slide structure.")

    if st.session_state.pptx_file:
        with open(st.session_state.pptx_file, "rb") as file:
            st.download_button(
                label="⬇️ Download Your Presentation",
                data=file,
                file_name=os.path.basename(st.session_state.pptx_file),
                mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                use_container_width=True
            )

# --- RIGHT COLUMN: TEMPLATE SELECTION ---
with right_col:
    st.subheader("2. Choose a Visual Template")
    
    previews = load_all_previews()
    if not previews:
        st.warning("No presentation themes could be generated!")
    else:
        template_cols = st.columns(3)
        theme_items = list(previews.items())
        for i, (theme_name, preview_image) in enumerate(theme_items):
            with template_cols[i % 3]:
                st.markdown(f'<div class="template-card">', unsafe_allow_html=True)
                st.image(preview_image, use_container_width=True)
                st.markdown('<div class="card-content">', unsafe_allow_html=True)
                st.markdown(f'<p class="template-name">{theme_name}</p>', unsafe_allow_html=True)
                if st.button("Select", key=f"select_{theme_name}"):
                    st.session_state.selected_template = theme_name
                    # No rerun needed here, selection is stored and checked on "Generate" click
                st.markdown('</div></div>', unsafe_allow_html=True)
    
    st.write("") # Spacer
    if st.session_state.selected_template:
        st.success(f"Selected Template: **{st.session_state.selected_template}**")
    else:
        st.info("Please select a template to proceed.")