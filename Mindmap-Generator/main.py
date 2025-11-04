# main.py

import streamlit as st
import os
from dotenv import load_dotenv
from utils import (
    search_academic_papers,
    get_paper_content,
    get_premium_paper_content,
    deep_analysis,
    generate_project_workflow,
)

# --- Initial Setup & Page Configuration ---
load_dotenv()

st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🧠",
    layout="wide",
)

# --- Custom CSS for UI Enhancement ---
def load_css():
    st.markdown("""
        <style>
            /* Main container styling */
            .stApp {
                background-color: #1a1a2e; /* Dark blue-purple background */
            }

            /* Custom card styling */
            .card {
                background-color: #2a2a3e;
                border-radius: 10px;
                padding: 20px;
                margin-bottom: 20px;
                box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
                border: 1px solid #4a4a5e;
            }

            /* Styling for Streamlit's expander */
            [data-testid="stExpander"] {
                background-color: #3a3a4e;
                border-radius: 10px !important;
                border: 1px solid #4a4a5e !important;
            }
            [data-testid="stExpander"] > details > summary {
                font-size: 1.1em;
                font-weight: bold;
                color: #e0e0ff;
            }

            /* Button styling */
            .stButton>button {
                border-radius: 20px;
                border: 2px solid #00BFFF;
                color: #00BFFF;
                background-color: transparent;
                padding: 10px 24px;
                font-weight: bold;
                transition: all 0.3s ease-in-out;
            }
            .stButton>button:hover {
                background-color: #00BFFF;
                color: white;
                border-color: #00BFFF;
                box-shadow: 0 0 15px #00BFFF;
            }
            .stButton>button:active {
                background-color: #009ACD !important;
                color: white !important;
            }

            /* Style for tabs */
            [data-baseweb="tab-list"] {
                background-color: #1a1a2e;
            }
        </style>
    """, unsafe_allow_html=True)

load_css()

# --- Session State Initialization ---
if "deep_analyses" not in st.session_state:
    st.session_state.deep_analyses = []
if "selected_paper_analysis" not in st.session_state:
    st.session_state.selected_paper_analysis = None
if "workflow" not in st.session_state:
    st.session_state.workflow = ""

# --- API Key Loading ---
scraper_api_key = os.getenv("SCRAPER_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")
s2_api_key = os.getenv("S2_API_KEY")

# --- UI Header ---
st.markdown("""
    <div class="card">
        <h1 style='text-align: center; color: #e0e0ff;'>🧠 AI Research Assistant & Project Planner</h1>
        <p style='text-align: center; color: #a0a0bf;'>Automate literature analysis and generate actionable project plans with suggested improvements.</p>
    </div>
""", unsafe_allow_html=True)


# --- Main App UI ---
if not all([scraper_api_key, groq_api_key, s2_api_key]):
    st.error("API keys not found. Please create a .env file and add your SCRAPER_API_KEY, GROQ_API_KEY, and S2_API_KEY.")
else:
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        topic = st.text_input("Enter your research topic:", key="topic_input", placeholder="e.g., AI in healthcare diagnostics")

        if st.button("Deep Search Top Papers", key="search_button"):
            st.session_state.deep_analyses = []
            st.session_state.selected_paper_analysis = None
            st.session_state.workflow = ""
            
            with st.spinner("Searching for academic papers..."):
                papers = search_academic_papers(topic, s2_api_key)

            if papers is not None and len(papers) >= 5:
                top_5_papers = papers[:5]
                analyses_data = []
                
                progress_bar = st.progress(0, text="Starting deep analysis...")
                for i, paper in enumerate(top_5_papers):
                    progress_text = f"Analyzing paper {i+1}/5: {paper['title'][:50]}..."
                    progress_bar.progress((i) / 5, text=progress_text)
                    
                    full_text = get_paper_content(paper['url'], scraper_api_key)
                    if len(full_text) < 500:
                        full_text = get_premium_paper_content(paper['url'], scraper_api_key)
                    
                    source_type = "Full Text" if len(full_text) >= 500 else "Abstract Only"
                    analysis_source_text = full_text if source_type == "Full Text" else paper.get('abstract', '')

                    analysis_result = deep_analysis(analysis_source_text, paper, groq_api_key)
                    analyses_data.append({"paper": paper, "analysis": analysis_result, "source": source_type})
                
                progress_bar.progress(1.0, text="Deep analysis complete!")
                st.session_state.deep_analyses = analyses_data
            else:
                st.warning("Could not find at least five relevant papers. Please try another topic.")
        st.markdown('</div>', unsafe_allow_html=True)


    # --- Step 2 & 3: Display Results in a Two-Column Layout ---
    if st.session_state.deep_analyses:
        col1, col2 = st.columns([1, 2]) # Left column is 1/3, Right column is 2/3

        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("📚 Select a Paper")
            titles = []
            for item in st.session_state.deep_analyses:
                icon = "✅" if item["source"] == "Full Text" else "⚠️"
                titles.append(f"{icon} {item['paper']['title']} ({item['paper']['year']})")
            
            selected_title_with_icon = st.radio(
                "Choose a paper to analyze:", 
                titles, 
                key="paper_selection",
                label_visibility="collapsed"
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Find the selected paper's full data
            selected_item = None
            for item in st.session_state.deep_analyses:
                icon = "✅" if item["source"] == "Full Text" else "⚠️"
                if f"{icon} {item['paper']['title']} ({item['paper']['year']})" == selected_title_with_icon:
                    selected_item = item
                    st.session_state.selected_paper_analysis = item["analysis"]
                    break

        with col2:
            if selected_item:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                
                # Create Tabs for Analysis and Workflow
                tab1, tab2 = st.tabs(["📄 Detailed Analysis", "🚀 Project Workflow"])

                with tab1:
                    expander_title = f"**{selected_item['paper']['title']}** - Citations: {selected_item['paper']['citationCount']}"
                    with st.expander(expander_title, expanded=True):
                        st.markdown(selected_item["analysis"], unsafe_allow_html=True)

                with tab2:
                    st.info("Click the button below to generate a new project workflow based on the detailed analysis.", icon="💡")
                    if st.button("Generate Project Workflow", key="workflow_button"):
                        with st.spinner("🤖 Generating actionable project workflow with improvements..."):
                            workflow_markdown = generate_project_workflow(st.session_state.selected_paper_analysis, groq_api_key)
                            st.session_state.workflow = workflow_markdown
                    
                    if st.session_state.workflow:
                        st.markdown(st.session_state.workflow, unsafe_allow_html=True)

                st.markdown('</div>', unsafe_allow_html=True)