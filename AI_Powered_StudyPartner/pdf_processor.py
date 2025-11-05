import fitz  # PyMuPDF
import io
from PIL import Image

def process_pdf_text(uploaded_file):
    """
    Extracts all text from an uploaded PDF file.
    """
    if uploaded_file is None:
        return ""
    try:
        uploaded_file.seek(0)
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        return f"Error processing PDF text: {e}"

def get_text_from_page(uploaded_file, page_num):
    """
    Extracts text from a specific page of an uploaded PDF file.
    """
    if uploaded_file is None:
        return "Error: No file uploaded."
    try:
        uploaded_file.seek(0)
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        if 0 < page_num <= len(doc):
            page = doc.load_page(page_num - 1) # pages are 0-indexed
            text = page.get_text()
            return text if text.strip() else "Error: No text found on this page."
        else:
            return f"Error: Invalid page number. The PDF has {len(doc)} pages."
    except Exception as e:
        return f"Error processing PDF page: {e}"

def extract_images_from_pdf(uploaded_file):
    """
    Extracts all images from an uploaded PDF file and returns them as a list of byte strings.
    """
    if uploaded_file is None:
        return []
    
    images = []
    try:
        uploaded_file.seek(0)
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            image_list = page.get_images(full=True)
            
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                images.append(image_bytes)
                
    except Exception as e:
        # You might want to log this error or handle it more gracefully
        print(f"Error extracting images from PDF: {e}")
        return []
        
    return images