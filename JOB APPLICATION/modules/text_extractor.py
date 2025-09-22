import streamlit as st
import PyPDF2
import docx
import requests
import os


def get_text_from_docx(file):
    """Extracts text from a .docx file."""
    try:
        doc = docx.Document(file)
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        return '\n'.join(full_text)
    except Exception as e:
        st.error(f"Error reading docx file: {e}")
        return None


def get_text_from_pdf(file):
    """Extracts text from a .pdf file."""
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
        return text
    except Exception as e:
        st.error(f"Error reading pdf file: {e}")
        return None


def ocr_space_file(file_content, file_extension, api_key):
    """Sends a file to OCR.space API for text extraction (handles 3-page free limit)."""
    try:
        payload = {
            'apikey': api_key,
            'language': 'eng',
        }
        files = {
            'file': (f'resume.{file_extension}', file_content, f'application/{file_extension}')
        }
        response = requests.post(
            'https://api.ocr.space/parse/image',
            data=payload,
            files=files
        )
        response.raise_for_status()
        result = response.json()

        # If OCR.space returns error due to page limit, still capture partial text
        if result.get('IsErroredOnProcessing'):
            error_msg = result.get('ErrorMessage')
            st.warning(f"OCR.space Warning: {error_msg}. Extracting available pages...")
            if "ParsedResults" in result:
                return " ".join([r['ParsedText'] for r in result['ParsedResults']])
            return None

        return " ".join([r['ParsedText'] for r in result['ParsedResults']])
    except requests.exceptions.RequestException as e:
        st.error(f"API Request Error: {e}")
        return None


def extract_text(uploaded_file):
    """Main function to extract text from an uploaded file."""
    if uploaded_file is None:
        return None

    file_extension = uploaded_file.name.split('.')[-1].lower()
    file_content = uploaded_file.getvalue()
    text = ""

    # Try direct extraction first
    if file_extension == 'pdf':
        text = get_text_from_pdf(uploaded_file)
    elif file_extension == 'docx':
        text = get_text_from_docx(uploaded_file)
    else:
        st.warning("Unsupported file type for direct extraction. Trying OCR...")
        text = None

    # If direct extraction fails or text is too short → fallback to OCR.space
    if not text or len(text.strip()) < 50:
        st.info("Direct text extraction failed or yielded minimal text. Attempting OCR with OCR.space...")
        api_key = os.getenv("OCR_SPACE_API_KEY")
        if not api_key:
            st.error("OCR.space API key is not configured.")
            return None
        text = ocr_space_file(file_content, file_extension, api_key)

    return text
