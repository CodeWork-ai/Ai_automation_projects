# app.py

import streamlit as st
from analyzer import run_full_analysis
import time
import csv
import io
import pandas as pd  # Import pandas for table display

# Import libraries for file handling
from PyPDF2 import PdfReader
import docx

# --- Page Configuration ---
st.set_page_config(
    page_title="Comprehensive Project Analyzer",
    page_icon="🚀",
    layout="wide"
)

# --- Sidebar ---
with st.sidebar:
    st.header("About This Tool")
    st.info(
        "This app uses a dual-pass LLM process to first extract high-level strategic insights from a project document, and then performs a detailed classification of individual requirements."
    )
    st.header("Supported File Types")
    st.markdown("- **.txt**\n- **.md**\n- **.pdf**\n- **.docx**")
    st.header("Export Options")
    st.markdown("- **.md** for Project Dashboard\n- **.csv** for Detailed Requirements")

# --- Helper Functions ---

def convert_overview_to_markdown(overview):
    """Converts the overview dictionary to a dashboard-style Markdown string."""
    md_string = "# Project Analysis Report\n\n"
    
    md_string += "## 🎯 Business Goals & Success Metrics\n\n"
    md_string += "### Business Goals\n"
    for item in overview.get("business_context_and_goals", []):
        md_string += f"- {item.get('point', 'N/A')}\n"
    md_string += "\n### Success Metrics\n"
    for item in overview.get("success_metrics", []):
        md_string += f"- {item.get('point', 'N/A')}\n"
        
    md_string += "\n## 👥 Stakeholders & Data Model\n\n"
    md_string += "### Stakeholders & User Needs\n"
    for item in overview.get("stakeholders_and_users", []):
        md_string += f"- **{item.get('role', 'N/A')}:** {item.get('needs', 'N/A')}\n"
    md_string += "\n### Key Data Requirements\n"
    for item in overview.get("data_requirements", []):
        md_string += f"- {item.get('point', 'N/A')}\n"

    md_string += "\n## 🔧 Technical Mapping\n\n"
    tech_map = overview.get("technical_mapping", {})
    if tech_map:
        md_string += "### Inferred Tech Stack\n"
        for key, value in tech_map.get('inferred_stack', {}).items():
            md_string += f"- **{key.replace('_', ' ').title()}:** {value}\n"
        md_string += "\n### Key System Components\n"
        for component in tech_map.get('components', []):
            md_string += f"- {component}\n"

    md_string += "\n## ⚠️ Risks & Constraints\n\n"
    md_string += "### Project Risks\n"
    for item in overview.get("risks_and_assumptions", []):
        md_string += f"- {item.get('point', 'N/A')}\n"
    md_string += "\n### Constraints\n"
    for item in overview.get("constraints", []):
        md_string += f"- {item.get('point', 'N/A')}\n"
        
    return md_string

def convert_specs_to_csv(specs):
    """Converts the detailed specifications to a CSV formatted string."""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Category', 'Subcategory', 'Requirement'])
    for category, items in specs.items():
        if items:
            for item in items:
                writer.writerow([category, item.get('subcategory', 'N/A'), item.get('requirement', '')])
    return output.getvalue()

def read_file(uploaded_file):
    """Reads the content of an uploaded file based on its type."""
    try:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        content = ""
        if file_extension in ['txt', 'md']:
            content = uploaded_file.getvalue().decode("utf-8")
        elif file_extension == 'pdf':
            pdf_reader = PdfReader(uploaded_file)
            for page in pdf_reader.pages:
                content += page.extract_text() + "\n"
        elif file_extension == 'docx':
            doc = docx.Document(uploaded_file)
            for para in doc.paragraphs:
                content += para.text + "\n"
        return content
    except Exception as e:
        st.error(f"Error reading file '{uploaded_file.name}': {e}")
        return None

# --- Main Page ---
st.title("🚀 Comprehensive Project Analyzer")
st.markdown("Transform unstructured project briefs into a structured, actionable dashboard.")

if 'input_text' not in st.session_state:
    st.session_state.input_text = ""

st.header("1. Provide Your Input")
st.write("Upload a document (.txt, .md, .pdf, .docx) and its content will appear below.")

uploaded_file = st.file_uploader(
    "Upload a project document",
    type=['txt', 'md', 'pdf', 'docx'],
    key="file_uploader"
)

if uploaded_file:
    extracted_text = read_file(uploaded_file)
    if extracted_text:
        st.session_state.input_text = extracted_text
        st.success(f"Successfully extracted text from '{uploaded_file.name}'. You can now start the analysis.")

client_input_area = st.text_area(
    "Review extracted text or paste your content here...",
    height=250,
    key="input_text"
)

st.header("2. Start the Analysis")
if st.button("Generate Full Analysis", type="primary"):
    if st.session_state.input_text:
        with st.spinner("Performing comprehensive analysis... This involves multiple LLM calls and may take a moment."):
            document_overview, detailed_specs = run_full_analysis(st.session_state.input_text)
        
        if "error" in document_overview:
            st.error(document_overview["error"])
        else:
            st.success("Analysis complete!")
            st.session_state.document_overview = document_overview
            st.session_state.detailed_specs = detailed_specs
            st.session_state.analysis_done = True
    else:
        st.warning("Please upload a file or paste some text to analyze.")

# --- Display Results and Download Buttons ---
if st.session_state.get('analysis_done', False):
    
    document_overview = st.session_state.document_overview
    detailed_specs = st.session_state.detailed_specs

    overview_tab, details_tab = st.tabs(["Project Dashboard 🚀", "Detailed Requirements Breakdown ⚙️"])

    with overview_tab:
        st.header("High-Level Project Dashboard")
        markdown_data = convert_overview_to_markdown(document_overview)
        st.download_button(
            label="⬇️ Download Dashboard as Markdown (.md)",
            data=markdown_data,
            file_name="project_dashboard.md",
            mime="text/markdown",
        )
        st.divider()

        # --- Section 1: Goals & Metrics ---
        st.subheader("🎯 Goals & Success Metrics")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Business Goals")
            for item in document_overview.get("business_context_and_goals", []):
                with st.container(border=True):
                    st.markdown(item['point'])
                    with st.expander("Source"):
                        for source in item.get('sources', []): st.info(f'"{source}"')
        with col2:
            st.markdown("#### Success Metrics")
            for item in document_overview.get("success_metrics", []):
                with st.container(border=True):
                    st.success(item['point'])
                    with st.expander("Source"):
                        for source in item.get('sources', []): st.info(f'"{source}"')
        
        st.divider()

        # --- Section 2: Users & Data ---
        st.subheader("👥 Stakeholders & Data Model")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Stakeholders & User Needs")
            stakeholder_data = document_overview.get("stakeholders_and_users", [])
            if stakeholder_data:
                df = pd.DataFrame(stakeholder_data)
                st.dataframe(df[['role', 'needs']], use_container_width=True, hide_index=True)
            else:
                st.markdown("_Not found_")
        with col2:
            st.markdown("#### Key Data Requirements")
            for item in document_overview.get("data_requirements", []):
                with st.container(border=True):
                    st.markdown(item['point'])
                    with st.expander("Source"):
                        for source in item.get('sources', []): st.info(f'"{source}"')

        st.divider()

        # --- Section 3: Technical Architecture ---
        st.subheader("🔧 Technical Mapping")
        tech_map = document_overview.get("technical_mapping", {})
        if tech_map:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Inferred Tech Stack")
                st.json(tech_map.get('inferred_stack', {}))
            with col2:
                st.markdown("#### Key System Components")
                for component in tech_map.get('components', []):
                    st.markdown(f"- {component}")
        else:
            st.markdown("_Not found_")

        st.divider()
        
        # --- Section 4: Risks & Constraints ---
        st.subheader("⚠️ Risks & Constraints")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Project Risks")
            for item in document_overview.get("risks_and_assumptions", []):
                with st.container(border=True):
                    st.warning(item['point'])
                    with st.expander("Source"):
                        for source in item.get('sources', []): st.info(f'"{source}"')
        with col2:
            st.markdown("#### Constraints")
            for item in document_overview.get("constraints", []):
                with st.container(border=True):
                    st.warning(item['point'])
                    with st.expander("Source"):
                        for source in item.get('sources', []): st.info(f'"{source}"')

    with details_tab:
        st.header("Detailed Requirements Classification")
        csv_data = convert_specs_to_csv(detailed_specs)
        st.download_button(
            label="⬇️ Download Requirements as CSV (.csv)",
            data=csv_data,
            file_name="detailed_requirements.csv",
            mime="text/csv",
        )
        st.divider()
        for category, items in detailed_specs.items():
            if items:
                icon = {"Functional": "⚙️", "Non-Functional": "🛡️", "Errors": "🔥"}.get(category, "❓")
                with st.expander(f"{icon} {category} Requirements ({len(items)})", expanded=True):
                    for item in items:
                        st.markdown(f"**Requirement:** `{item['requirement']}`")
                        st.info(f"**Subcategory:** {item['subcategory']}")
                        st.markdown("---")