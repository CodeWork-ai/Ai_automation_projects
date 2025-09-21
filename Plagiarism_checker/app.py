import streamlit as st
import hashlib
import os
import PyPDF2
from docx import Document
from dotenv import load_dotenv
from groq import Groq
from serpapi import GoogleSearch
from playwright.async_api import async_playwright
import asyncio
import re
import nltk
import random

# --- Initial Setup ---
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    with st.spinner("First-time setup: Downloading language model..."):
        nltk.download('punkt')
    st.experimental_rerun()

load_dotenv()

# --- Page Configuration ---
st.set_page_config(
    page_title="Content Analyzer Toolkit",
    page_icon="🔎",
    layout="wide"
)

st.title("🔎 Content Analyzer Toolkit")

# --- API Key Initialization (only for Plagiarism Checker) ---
groq_api_key = os.getenv("GROQ_API_KEY")
serpapi_api_key = os.getenv("SERPAPI_API_KEY")

# --- Core Helper Functions ---
def read_file_content(file):
    try:
        ext = file.name.split('.')[-1].lower()
        if ext == 'txt': return file.getvalue().decode("utf-8")
        if ext == 'docx':
            doc = Document(file)
            return "\n".join([para.text for para in doc.paragraphs])
        if ext == 'pdf':
            reader = PyPDF2.PdfReader(file)
            return "".join(page.extract_text() or "" for page in reader.pages)
    except Exception as e:
        st.error(f"Error reading file {file.name}: {e}")
    return ""

# --- Mode Selection Switch ---
st.sidebar.title("Select Analysis Mode")
mode = st.sidebar.radio(
    "Choose what you want to do:",
    ("Plagiarism Checker", "Duplicate File Checker")
)

# ==============================================================================
# === MODE 1: PLAGIARISM CHECKER
# ==============================================================================
if mode == "Plagiarism Checker":
    st.header("Plagiarism & Web Source Checker")

    if not groq_api_key or not serpapi_api_key:
        st.error("GROQ_API_KEY and SERPAPI_API_KEY must be set in your .env file for this mode.")
        st.stop()

    client = Groq(api_key=groq_api_key)

    st.subheader("Option 1: Paste Your Content")
    pasted_text = st.text_area("Enter text to check for plagiarism.", height=250, placeholder="Paste your article or essay here...")

    st.divider()

    st.subheader("Option 2: Upload a Single Document")
    uploaded_file = st.file_uploader("Or, upload a TXT, PDF, or DOCX file.", type=['txt', 'pdf', 'docx'])

    analyze_button = st.button("Analyze for Plagiarism", type="primary")

    def search_web_for_plagiarism(text):
        st.write("Breaking content into sentences...")
        sentences = nltk.sent_tokenize(text)
        sentences_to_check = [s for s in sentences if len(s.split()) > 8]
        if not sentences_to_check: return []
        sample_size = min(5, len(sentences_to_check))
        sentences_sample = random.sample(sentences_to_check, sample_size)
        found_sources = []
        progress_bar = st.progress(0, text="Searching web for selected sentences...")
        for i, sentence in enumerate(sentences_sample):
            progress_bar.progress((i + 1) / len(sentences_sample), text=f"Searching: \"{sentence[:50]}...\"")
            try:
                search = GoogleSearch({"q": f'"{sentence}"', "api_key": serpapi_api_key})
                results = search.get_dict()
                if 'organic_results' in results and results['organic_results']:
                    res = results['organic_results'][0]
                    found_sources.append({"sentence": sentence, "url": res['link'], "source_title": res['title']})
            except Exception as e:
                st.warning(f"SerpApi search failed. Error: {e}")
        progress_bar.empty()
        return found_sources

    def run_plagiarism_analysis(content, source_name):
        st.header(f"Web Source Analysis for: *{source_name}*")
        found_sources = search_web_for_plagiarism(content)
        if not found_sources:
            st.success("✅ **Content Appears to be Unique**")
            st.balloons()
            st.write("No direct matches were found on the web for the sampled sentences.")
            return
        st.error("🚨 **Potential Plagiarism Detected!**")
        st.write("The following sentences from your content were found on other websites.")
        st.subheader("Found Sentences and Their Online Sources")
        for source in found_sources:
            with st.container(border=True):
                st.markdown(f"**Sentence:** *\"{source['sentence']}\"*")
                st.markdown(f"**Found At:** [{source['source_title']}]({source['url']})")

    if analyze_button:
        if pasted_text.strip():
            run_plagiarism_analysis(pasted_text, "Pasted Content")
        elif uploaded_file:
            file_content = read_file_content(uploaded_file)
            if file_content:
                run_plagiarism_analysis(file_content, f"File: `{uploaded_file.name}`")
        else:
            st.warning("Please paste text or upload a file to analyze.")

# ==============================================================================
# === MODE 2: DUPLICATE FILE CHECKER
# ==============================================================================
elif mode == "Duplicate File Checker":
    st.header("Duplicate File Checker")
    st.info("Upload two or more files to find which ones are exact duplicates of each other.")

    uploaded_files = st.file_uploader(
        "Upload multiple TXT, PDF, or DOCX files.",
        accept_multiple_files=True,
        type=['txt', 'pdf', 'docx']
    )

    analyze_duplicates_button = st.button("Find Duplicates", type="primary")

    def handle_duplicate_check(files):
        if len(files) < 2:
            st.warning("Please upload at least two files to compare.")
            return

        with st.spinner("Reading files and calculating hashes..."):
            hashes = {}
            for file in files:
                content = read_file_content(file)
                if content:
                    content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
                    if content_hash in hashes:
                        hashes[content_hash].append(file.name)
                    else:
                        hashes[content_hash] = [file.name]

        st.subheader("Duplicate Analysis Report")
        found_duplicates = False
        for content_hash, filenames in hashes.items():
            if len(filenames) > 1:
                found_duplicates = True
                with st.container(border=True):
                    st.error(f"Found {len(filenames) - 1} duplicate(s) for the file `{filenames[0]}`:")
                    for duplicate_name in filenames[1:]:
                        st.write(f"-> **`{duplicate_name}`**")

        if not found_duplicates:
            st.success("✅ **No duplicate files were found.** All uploaded files have unique content.")

    if analyze_duplicates_button:
        handle_duplicate_check(uploaded_files)
