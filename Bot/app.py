import streamlit as st
import os
import json
import re
import tempfile
import hashlib
import warnings
import logging
import traceback
import shutil
from dotenv import load_dotenv
from pathlib import Path
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from firecrawl import Firecrawl
from urllib.parse import urlparse, urljoin
import requests
from datetime import datetime
import time

# Suppress warnings
warnings.filterwarnings("ignore")
logging.getLogger("streamlit").setLevel(logging.CRITICAL)

# Enhanced logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === LlamaIndex imports ===
from llama_index.core import (
    VectorStoreIndex, StorageContext, load_index_from_storage,
    Document as LlamaDocument, Settings
)
from llama_index.core.schema import TextNode, BaseNode
from llama_index.readers.file import PyMuPDFReader as PDFReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer

# Vector store imports
from llama_index.core.vector_stores import SimpleVectorStore

# BM25 import
from llama_index.retrievers.bm25 import BM25Retriever

# === CRITICAL FIX: Load and validate environment variables ===
try:
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")
    
    if not OPENAI_API_KEY:
        st.error("❌ OPENAI_API_KEY not found!")
        st.markdown("""
        ### How to fix this:
        
        **Option 1: Create a .env file**
        1. Create a file named `.env` in your project directory
        2. Add this line: `OPENAI_API_KEY=your_openai_api_key_here`
        3. Replace `your_openai_api_key_here` with your actual OpenAI API key
        4. Restart the Streamlit app
        
        **Option 2: Set environment variable**
        - **Windows**: `setx OPENAI_API_KEY "your_openai_api_key_here"`
        - **macOS/Linux**: `export OPENAI_API_KEY="your_openai_api_key_here"`
        
        **Get your OpenAI API key at**: [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)
        """)
        st.stop()
    
    if not FIRECRAWL_API_KEY:
        st.error("❌ FIRECRAWL_API_KEY not found!")
        st.markdown("""
        ### How to add Firecrawl API Key:
        
        1. Get your API key from [https://www.firecrawl.dev/](https://www.firecrawl.dev/)
        2. Add this line to your `.env` file: `FIRECRAWL_API_KEY=your_firecrawl_api_key_here`
        3. Replace `your_firecrawl_api_key_here` with your actual Firecrawl API key
        4. Restart the Streamlit app
        """)
        st.stop()
    
    # Set environment variables
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    os.environ["FIRECRAWL_API_KEY"] = FIRECRAWL_API_KEY
    
except Exception as e:
    st.error(f"❌ Error loading environment variables: {str(e)}")
    st.markdown("Please ensure you have a `.env` file with your API keys")
    st.stop()

# === Config ===
COMPANY_EMAIL = "sales@codework.ai"
COMPANY_PHONE = "+91 75989 81500"

# === Database imports and setup ===
from database import (
    Document, Chat, Message, 
    SessionLocal, init_db
)
from sqlalchemy.orm import Session
from contextlib import contextmanager

# Initialize database
try:
    init_db()
except Exception as e:
    st.error(f"❌ Database initialization failed: {str(e)}")
    st.info("Please ensure your database.py file is properly configured")
    st.stop()

@contextmanager
def get_db_context():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# === Helper functions ===
def get_file_hash(uploaded_file):
    """Generate a unique hash for the uploaded file"""
    file_content = uploaded_file.read()
    uploaded_file.seek(0)
    return hashlib.md5(file_content).hexdigest()

def get_url_hash(url):
    """Generate a unique hash for a URL"""
    return hashlib.md5(url.encode()).hexdigest()

def save_uploaded_file(db: Session, file_content: bytes, filename: str, file_hash: str, content_type: str = 'pdf'):
    """Save uploaded file to database"""
    db_document = Document(
        filename=filename,
        file_hash=file_hash,
        content_type=content_type,
        file_data=file_content
    )
    db.add(db_document)
    db.commit()
    db.refresh(db_document)
    return db_document

def save_website_data(db: Session, url: str, content: str, url_hash: str):
    """Save website data to database"""
    db_document = Document(
        filename=f"Website: {url}",
        file_hash=url_hash,
        content_type='website',
        file_data=content.encode('utf-8')
    )
    db.add(db_document)
    db.commit()
    db.refresh(db_document)
    return db_document

def get_document_by_hash(db: Session, file_hash: str):
    """Retrieve document from database by hash"""
    return db.query(Document).filter(Document.file_hash == file_hash).first()

def create_chat(db: Session, document_id: int, chat_id: int):
    """Create a new chat in the database"""
    db_chat = Chat(document_id=document_id, chat_id=chat_id)
    db.add(db_chat)
    db.commit()
    db.refresh(db_chat)
    return db_chat

def save_message(db: Session, chat_id: int, role: str, content: str):
    """Save a message to the database"""
    db_message = Message(chat_id=chat_id, role=role, content=content)
    db.add(db_message)
    db.commit()
    db.refresh(db_message)
    return db_message

def get_chat_messages(db: Session, chat_id: int, document_id: int = None):
    """Get all messages for a chat"""
    try:
        if document_id:
            chat = db.query(Chat).filter(
                Chat.chat_id == chat_id,
                Chat.document_id == document_id
            ).first()
        else:
            chat = db.query(Chat).filter(Chat.chat_id == chat_id).first()
        
        if chat:
            messages = db.query(Message).filter(
                Message.chat_id == chat.id
            ).order_by(Message.timestamp).all()
            
            logger.info(f"Retrieved {len(messages)} messages for chat_id={chat_id}, db_chat_id={chat.id}")
            return messages
        else:
            logger.warning(f"No chat found for chat_id={chat_id}")
            return []
            
    except Exception as e:
        logger.error(f"Error retrieving messages for chat_id={chat_id}: {str(e)}")
        return []

# === FIXED: Firecrawl Integration ===
def crawl_website(url, max_pages=50):
    """Crawl website using Firecrawl API - FIXED VERSION"""
    try:
        firecrawl = Firecrawl(api_key=FIRECRAWL_API_KEY)
        
        # Validate URL
        parsed_url = urlparse(url)
        if not parsed_url.scheme:
            url = "https://" + url
        
        st.info(f"🔥 Starting website crawl for: {url}")
        st.info(f"📄 Maximum pages to crawl: {max_pages}")
        
        # Start the crawl job using start_crawl method
        crawl_job = firecrawl.start_crawl(
            url=url,
            limit=max_pages,
            scrape_options={
                "formats": ["markdown"],
                "only_main_content": True,
                "include_html": False
            }
        )
        
        # Check if crawl job started successfully
        if not crawl_job or not hasattr(crawl_job, 'id'):
            st.error("Failed to start crawl job. Please check the URL and try again.")
            return None, None
        
        st.info(f"🚀 Crawl job started with ID: {crawl_job.id}")
        st.info("⏳ Waiting for crawl to complete...")
        
        # Poll for completion
        max_wait_time = 300  # 5 minutes
        poll_interval = 10   # Check every 10 seconds
        start_time = time.time()
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        while time.time() - start_time < max_wait_time:
            # Get crawl status
            status = firecrawl.get_crawl_status(crawl_job.id)
            
            if hasattr(status, 'status'):
                current_status = status.status
                completed = getattr(status, 'completed', 0)
                total = getattr(status, 'total', max_pages)
                
                # Update progress
                if total > 0:
                    progress = min(completed / total, 1.0)
                    progress_bar.progress(progress)
                    status_text.text(f"Status: {current_status} - {completed}/{total} pages completed")
                
                if current_status == "completed":
                    st.success("✅ Crawl completed successfully!")
                    
                    # Get the crawled data
                    crawl_data = getattr(status, 'data', [])
                    
                    if not crawl_data:
                        st.error("No pages found in crawl results.")
                        return None, None
                    
                    # Process the crawled pages
                    combined_content = ""
                    page_info = []
                    
                    for i, page in enumerate(crawl_data):
                        # Handle different possible data structures
                        if hasattr(page, 'metadata'):
                            page_url = getattr(page.metadata, 'sourceURL', f'Page {i+1}')
                            page_title = getattr(page.metadata, 'title', 'Untitled')
                        elif isinstance(page, dict):
                            metadata = page.get('metadata', {})
                            page_url = metadata.get('sourceURL', f'Page {i+1}')
                            page_title = metadata.get('title', 'Untitled')
                        else:
                            page_url = f'Page {i+1}'
                            page_title = 'Untitled'
                        
                        # Get page content
                        page_content = ""
                        if hasattr(page, 'markdown') and page.markdown:
                            page_content = page.markdown
                        elif hasattr(page, 'content') and page.content:
                            page_content = page.content
                        elif isinstance(page, dict):
                            page_content = page.get('markdown', page.get('content', ''))
                        
                        if page_content and page_content.strip():
                            page_info.append({
                                'url': page_url,
                                'title': page_title,
                                'content_length': len(page_content)
                            })
                            
                            combined_content += f"\n\n=== PAGE: {page_title} ({page_url}) ===\n\n"
                            combined_content += page_content
                    
                    # Clear progress indicators
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Display crawl summary
                    st.success(f"📊 Successfully crawled {len(page_info)} pages with content!")
                    
                    with st.expander("🔍 Crawled Pages Summary", expanded=False):
                        for info in page_info[:10]:  # Show first 10 pages
                            st.text(f"• {info['title']} ({info['content_length']} chars)")
                        if len(page_info) > 10:
                            st.text(f"... and {len(page_info) - 10} more pages")
                    
                    return combined_content, page_info
                
                elif current_status == "failed":
                    st.error("❌ Crawl job failed.")
                    error_msg = getattr(status, 'error', 'Unknown error')
                    st.error(f"Error details: {error_msg}")
                    return None, None
                
                # Continue polling
                time.sleep(poll_interval)
            else:
                st.warning("Unable to get crawl status. Retrying...")
                time.sleep(poll_interval)
        
        # Timeout reached
        progress_bar.empty()
        status_text.empty()
        st.error(f"⏰ Crawl timeout reached ({max_wait_time}s). The crawl may still be running.")
        return None, None
        
    except Exception as e:
        logger.error(f"Error crawling website: {str(e)}")
        st.error(f"❌ Failed to crawl website: {str(e)}")
        
        # Show more detailed error information
        with st.expander("🔍 Error Details"):
            st.code(traceback.format_exc())
        
        return None, None

def handle_url_input():
    """Handle URL input and crawling"""
    
    if "current_url_info" not in st.session_state:
        st.session_state.current_url_info = None
    if "url_processing_complete" not in st.session_state:
        st.session_state.url_processing_complete = False
    
    st.markdown("### 🌐 Website Crawler")
    
    with st.form("url_crawl_form", clear_on_submit=True):
        url_input = st.text_input(
            "Enter website URL to crawl",
            placeholder="https://example.com",
            help="Enter the base URL of the website you want to crawl"
        )
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            max_pages = st.slider("Max pages to crawl", min_value=5, max_value=100, value=20)
        
        with col2:
            crawl_button = st.form_submit_button("🔥 Crawl Website", use_container_width=True)
        
        with col3:
            if st.form_submit_button("🗑️ Clear", use_container_width=True):
                st.session_state.current_url_info = None
                st.session_state.url_processing_complete = False
                
                keys_to_clear = [
                    'current_url_hash', 'current_chat_id', 'chat_history', 'chat_engine', 
                    'rag_chain', 'retriever_chain', 'index', 'bm25_retriever', 'force_rebuild'
                ]
                for key in keys_to_clear:
                    if key in st.session_state:
                        del st.session_state[key]
                
                st.success("🗑️ Website data cleared! You can now crawl a new website.")
                st.rerun()
        
        if crawl_button and url_input.strip():
            try:
                url = url_input.strip()
                url_hash = get_url_hash(url)
                
                # Check if this is a new URL
                if (st.session_state.current_url_info is None or 
                    st.session_state.current_url_info["hash"] != url_hash):
                    
                    # Clear session state for new URL
                    keys_to_clear = [
                        'current_chat_id', 'chat_history', 'chat_engine', 
                        'rag_chain', 'retriever_chain', 'index', 'bm25_retriever'
                    ]
                    for key in keys_to_clear:
                        if key in st.session_state:
                            del st.session_state[key]
                    
                    # Crawl the website
                    combined_content, page_info = crawl_website(url, max_pages)
                    
                    if combined_content:
                        # Save to database
                        with get_db_context() as db:
                            existing_doc = get_document_by_hash(db, url_hash)
                            if not existing_doc:
                                save_website_data(db, url, combined_content, url_hash)
                        
                        # Store URL info in session state
                        st.session_state.current_url_info = {
                            "hash": url_hash,
                            "url": url,
                            "pages": len(page_info) if page_info else 0,
                            "content_length": len(combined_content)
                        }
                        st.session_state.current_file_hash = url_hash
                        st.session_state.url_processing_complete = False
                        st.session_state.force_rebuild = True
                        
                        st.success(f"✅ Website crawled successfully: {len(page_info)} pages")
                        st.rerun()
                    else:
                        st.error("Failed to crawl website. Please check the URL and try again.")
                else:
                    st.info("🌐 This website is already crawled and processed.")
                    
            except Exception as e:
                st.error(f"Crawling failed: {str(e)}")
                return None, None
    
    # Display current URL status
    if st.session_state.current_url_info:
        url_info = st.session_state.current_url_info
        st.success(f"🌐 **Current Website**: {url_info['url']}")
        st.info(f"📊 Pages crawled: {url_info['pages']} | Content size: {url_info['content_length'] / 1024 / 1024:.2f} MB")
        return url_info["hash"], url_info["url"]
    
    return None, None

# === ENHANCED: File upload with URL option ===
def handle_document_input():
    """Enhanced document input handling - PDF upload OR URL crawling"""
    
    # Initialize session state
    if "current_file_info" not in st.session_state:
        st.session_state.current_file_info = None
    if "current_url_info" not in st.session_state:
        st.session_state.current_url_info = None
    if "processing_complete" not in st.session_state:
        st.session_state.processing_complete = False
    
    # Document input tabs
    tab1, tab2 = st.tabs(["📁 Upload PDF", "🌐 Crawl Website"])
    
    with tab1:
        file_hash, filename = handle_file_upload()
        if file_hash:
            return file_hash, f"PDF: {filename}"
    
    with tab2:
        url_hash, url = handle_url_input()
        if url_hash:
            return url_hash, f"Website: {url}"
    
    return None, None

def handle_file_upload():
    """Enhanced file upload with proper reset capability"""
    
    st.markdown("### 📁 Upload Document")

    with st.form("pdf_upload_form", clear_on_submit=True):
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=['pdf'],
            help="Upload a PDF document to start chatting",
            label_visibility="visible"
        )
        
        col1, col2 = st.columns([3, 1])
        with col1:
            submit_button = st.form_submit_button("📤 Upload & Process", use_container_width=True)
        with col2:
            if st.form_submit_button("🗑️ Clear", use_container_width=True):
                st.session_state.current_file_info = None
                st.session_state.processing_complete = False
                
                keys_to_clear = [
                    'current_file_hash', 'current_chat_id', 'chat_history', 'chat_engine', 
                    'rag_chain', 'retriever_chain', 'index', 'bm25_retriever', 'force_rebuild'
                ]
                for key in keys_to_clear:
                    if key in st.session_state:
                        del st.session_state[key]
                
                st.success("🗑️ Document cleared! You can now upload a new file.")
                st.rerun()
        
        if submit_button and uploaded_file is not None:
            try:
                file_hash = get_file_hash(uploaded_file)
                file_content = uploaded_file.read()
                uploaded_file.seek(0)
                
                if (st.session_state.current_file_info is None or 
                    st.session_state.current_file_info["hash"] != file_hash):
                    
                    keys_to_clear = [
                        'current_chat_id', 'chat_history', 'chat_engine', 
                        'rag_chain', 'retriever_chain', 'index', 'bm25_retriever'
                    ]
                    for key in keys_to_clear:
                        if key in st.session_state:
                            del st.session_state[key]
                    
                    with get_db_context() as db:
                        existing_doc = get_document_by_hash(db, file_hash)
                        if not existing_doc:
                            save_uploaded_file(db, file_content, uploaded_file.name, file_hash)
                    
                    st.session_state.current_file_info = {
                        "hash": file_hash,
                        "name": uploaded_file.name,
                        "size": len(file_content)
                    }
                    st.session_state.current_file_hash = file_hash
                    st.session_state.processing_complete = False
                    st.session_state.force_rebuild = True
                    
                    st.success(f"✅ File uploaded: {uploaded_file.name}")
                    st.rerun()
                else:
                    st.info("📄 This file is already uploaded and processed.")
                    
            except Exception as e:
                st.error(f"Upload failed: {str(e)}")
                return None, None
    
    if st.session_state.current_file_info:
        file_info = st.session_state.current_file_info
        st.success(f"📄 **Current Document**: {file_info['name']} ({file_info['size'] / 1024 / 1024:.2f} MB)")
        return file_info["hash"], file_info["name"]
    else:
        st.info("👆 Please upload a PDF document to start chatting!")
    
    return None, None

# === Enhanced text preprocessing ===
def preprocess_pdf_text(text):
    """Enhanced text preprocessing"""
    text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = re.sub(r'^\s*[•·\-\*]\s*', '- ', text, flags=re.MULTILINE)
    return text.strip()

def preprocess_website_text(text):
    """Preprocessing for website content"""
    # Clean up markdown-style content
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    text = re.sub(r'\s+', ' ', text)
    
    # Clean up common web artifacts
    text = re.sub(r'\[.*?\]\(.*?\)', '', text)  # Remove markdown links
    text = re.sub(r'#{1,6}\s*', '', text)  # Remove markdown headers
    text = re.sub(r'\*{1,2}([^*]+)\*{1,2}', r'\1', text)  # Remove bold/italic
    
    return text.strip()

# === Rest of the code remains the same... ===
def load_or_build_index(file_hash, force_rebuild=False):
    """Load or build index with force rebuild option - Enhanced for websites"""
    with get_db_context() as db:
        document = get_document_by_hash(db, file_hash)
        if not document:
            st.error("Document not found in database")
            st.stop()

    try:
        # Initialize OpenAIEmbedding without the 'proxies' parameter
        # The 'proxies' parameter is not supported by LlamaIndex's OpenAIEmbedding class
        # Removed proxies parameter to fix "init() got an unexpected keyword argument 'proxies'" error
        Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    except Exception as e:
        st.error(f"❌ Failed to initialize OpenAI embeddings: {str(e)}")
        st.info("Please check your OPENAI_API_KEY is valid and has sufficient credits")
        st.stop()
    
    persist_dir = f"vector_stores/store_{file_hash}"
    os.makedirs(persist_dir, exist_ok=True)
    nodes_file = os.path.join(persist_dir, "nodes.json")
    
    if (force_rebuild or 
        not os.path.exists(os.path.join(persist_dir, "index_store.json")) or
        not os.path.exists(nodes_file)):
        
        if document.content_type == 'website':
            st.info("🔧 Building fresh index for crawled website...")
        else:
            st.info("🔧 Building fresh index for uploaded document...")
        
        try:
            vector_store = SimpleVectorStore()
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
        except Exception as e:
            st.error(f"Error initializing vector store: {str(e)}")
            st.stop()

        try:
            with get_db_context() as db:
                document = get_document_by_hash(db, file_hash)
                
                if document.content_type == 'website':
                    # Process website content
                    combined_text = document.file_data.decode('utf-8')
                    combined_text = preprocess_website_text(combined_text)
                    
                    st.info(f"🌐 Processing website: {document.filename}")
                    
                else:
                    # Process PDF
                    temp_pdf = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
                    try:
                        temp_pdf.write(document.file_data)
                        temp_pdf.flush()
                        temp_pdf.close()
                        
                        loader = PDFReader()
                        raw_docs = loader.load_data(temp_pdf.name)
                    finally:
                        try:
                            os.unlink(temp_pdf.name)
                        except Exception:
                            pass
            
                    if not raw_docs:
                        st.error("No content found in PDF file.")
                        st.stop()
                    
                    combined_text = "\n\n".join([doc.text for doc in raw_docs])
                    combined_text = preprocess_pdf_text(combined_text)
                    
                    st.info(f"📄 Processing document: {document.filename}")
                
                # Content verification
                with st.expander("🔍 Content Preview", expanded=False):
                    preview_text = combined_text[:2000] + "..." if len(combined_text) > 2000 else combined_text
                    st.text(preview_text)
                
                logger.info(f"📄 Processing {len(combined_text)} characters from {document.filename}")
                
                if not combined_text.strip():
                    st.error("Content appears to be empty or unreadable.")
                    st.stop()

                # Optimized chunking
                if document.content_type == 'website':
                    chunk_size = 500  # Larger chunks for website content
                    chunk_overlap = 100
                else:
                    chunk_size = 300  # Smaller chunks for PDF
                    chunk_overlap = 60
                
                splitter = SentenceSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    separator="\n\n"
                )
                
                st.info(f"🔧 Chunking: size={chunk_size}, overlap={chunk_overlap}")
                
                enhanced_doc = LlamaDocument(
                    text=combined_text,
                    metadata={
                        "filename": document.filename, 
                        "file_hash": file_hash,
                        "source": document.content_type,
                        "processed_at": datetime.now().isoformat()
                    }
                )
                
                nodes = splitter.get_nodes_from_documents([enhanced_doc])

                if not nodes:
                    st.error("Failed to create document chunks.")
                    st.stop()

                logger.info(f"🧩 Created {len(nodes)} chunks")
                st.info(f"📊 Created {len(nodes)} text chunks")
                
                with st.expander("🧩 Chunk Samples", expanded=False):
                    for i, node in enumerate(nodes[:3]):
                        content = node.get_content()
                        st.text(f"Chunk {i+1}: {content[:200]}...")

                st.info(f"🔄 Creating embeddings for {len(nodes)} chunks...")
                
                try:
                    index = VectorStoreIndex(
                        nodes=nodes,
                        storage_context=storage_context,
                        show_progress=True
                    )
                except Exception as e:
                    st.error(f"❌ Failed to create vector index: {str(e)}")
                    st.info("This might be due to OpenAI API issues or insufficient credits")
                    st.stop()
                
                # Persist
                index.storage_context.persist(persist_dir=persist_dir)
                
                # Save nodes for BM25
                nodes_data = []
                for node in nodes:
                    node_dict = {
                        'text': node.get_content(),
                        'id': node.node_id,
                        'metadata': node.metadata
                    }
                    nodes_data.append(node_dict)
                
                with open(nodes_file, 'w', encoding='utf-8') as f:
                    json.dump(nodes_data, f, ensure_ascii=False, indent=2)

                st.success(f"✅ Fresh index created with {len(nodes)} chunks!")
                
                bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=10)
                st.success("✅ BM25 retriever created!")
                
        except Exception as e:
            st.error(f"Error processing content: {str(e)}")
            st.stop()
            
    else:
        # Load existing
        st.info("Loading existing vector store...")
        try:
            storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
            index = load_index_from_storage(storage_context)
            
            with open(nodes_file, 'r', encoding='utf-8') as f:
                nodes_data = json.load(f)
            
            nodes = []
            for node_data in nodes_data:
                node = TextNode(text=node_data['text'], id_=node_data['id'])
                if 'metadata' in node_data:
                    node.metadata = node_data['metadata']
                nodes.append(node)
            
            bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=10)
            st.success(f"✅ Loaded existing index with {len(nodes)} chunks!")
        except Exception as e:
            st.warning(f"Failed to load existing index: {str(e)}. Building fresh one...")
            return load_or_build_index(file_hash, force_rebuild=True)

    return index, bm25_retriever

# === Enhanced Context Chat Engine ===
def create_chat_engine(_index):
    """Create context-aware chat engine with enhanced prompts"""
    try:
        memory = ChatMemoryBuffer.from_defaults(token_limit=8000)
        
        # Enhanced system prompt for both PDFs and websites
        system_prompt = (
            "You are an AI assistant that helps users understand documents and websites. "
            "Use ONLY the provided context to give detailed, accurate responses. "
            "When the context comes from a website, mention relevant page titles or sections when helpful. "
            "When the context comes from a PDF document, reference specific sections when available. "
            "Never make up information that is not in the context. "
            "If asked for more details about a topic, provide comprehensive information from the context. "
            "If the user asks about specific pages or sections of a website, try to identify and reference them. "
            "Be helpful, thorough, and cite specific parts of the content when relevant."
        )
        
        chat_engine = ContextChatEngine.from_defaults(
            retriever=_index.as_retriever(similarity_top_k=8),
            memory=memory,
            system_prompt=system_prompt,
        )
        
        return chat_engine
    except Exception as e:
        logger.error(f"Error creating chat engine: {str(e)}")
        st.error(f"❌ Failed to create chat engine: {str(e)}")
        return None

# === Chat Management Functions (unchanged) ===
def save_chat_history(chat_id, history, file_hash):
    try:
        with get_db_context() as db:
            document = get_document_by_hash(db, file_hash)
            if not document:
                return
            
            chat = db.query(Chat).filter(
                Chat.document_id == document.id,
                Chat.chat_id == chat_id
            ).first()
            
            if not chat:
                chat = create_chat(db, document.id, chat_id)
            
            db.query(Message).filter(Message.chat_id == chat.id).delete()
            
            for msg in history:
                role = "assistant" if isinstance(msg, AIMessage) else "user"
                save_message(db, chat.id, role, msg.content)
                
    except Exception as e:
        logger.error(f"Failed to save chat history: {str(e)}")

def load_chat_history(chat_id, file_hash):
    try:
        with get_db_context() as db:
            document = get_document_by_hash(db, file_hash)
            if not document:
                logger.warning(f"Document not found for hash: {file_hash}")
                return [AIMessage(content="Hi! I'm your document assistant. How can I help you today?")]
            
            chat = db.query(Chat).filter(
                Chat.document_id == document.id,
                Chat.chat_id == chat_id
            ).first()
            
            if not chat:
                logger.warning(f"No chat found for chat_id={chat_id}, document_id={document.id}")
                
                logger.info(f"Auto-creating missing chat {chat_id}")
                try:
                    chat = create_chat(db, document.id, chat_id)
                    logger.info(f"✅ Successfully created chat {chat_id}")
                except Exception as e:
                    logger.error(f"Failed to create chat {chat_id}: {str(e)}")
                
                return [AIMessage(content="Hi! I'm your document assistant. How can I help you today?")]
            
            messages = get_chat_messages(db, chat_id, document.id)
            
            if messages:
                chat_history = []
                for msg in messages:
                    if msg.role == "assistant":
                        chat_history.append(AIMessage(content=msg.content))
                    else:
                        chat_history.append(HumanMessage(content=msg.content))
                
                logger.info(f"✅ Loaded {len(chat_history)} messages for chat {chat_id}")
                return chat_history
            else:
                logger.info(f"Chat {chat_id} exists but has no messages")
                return [AIMessage(content="Hi! I'm your document assistant. How can I help you today?")]
            
    except Exception as e:
        logger.error(f"❌ Failed to load chat history for chat_id={chat_id}: {str(e)}")
        logger.error(f"Exception details: {traceback.format_exc()}")
        return [AIMessage(content="Hi! I'm your document assistant. How can I help you today?")]

def get_available_chats(file_hash):
    try:
        with get_db_context() as db:
            document = get_document_by_hash(db, file_hash)
            if not document:
                return []
            
            chats = db.query(Chat).filter(Chat.document_id == document.id).all()
            return [chat.chat_id for chat in chats]
    except Exception:
        return []

# === Main Streamlit App ===
def main():
    st.set_page_config(page_title="Document AI Assistant", page_icon="🤖", layout="wide")
    st.title("🤖 Document AI Assistant - RAG System with Website Crawler")
    
    # Display API key status
    api_status_col1, api_status_col2 = st.columns(2)
    
    with api_status_col1:
        if OPENAI_API_KEY:
            st.success(f"✅ OpenAI API Key loaded")
        else:
            st.error("❌ OpenAI API Key not found")
    
    with api_status_col2:
        if FIRECRAWL_API_KEY:
            st.success(f"✅ Firecrawl API Key loaded")
        else:
            st.error("❌ Firecrawl API Key not found")
    
    if not OPENAI_API_KEY or not FIRECRAWL_API_KEY:
        return

    # ENHANCED: Handle both PDF upload and website crawling
    file_hash, source_name = handle_document_input()

    if file_hash is None:
        st.markdown("""
        **🚀 Enhanced Features:**
        - **📁 PDF Upload**: Upload and chat with PDF documents
        - **🌐 Website Crawler**: Crawl entire websites using Firecrawl
        - **🤖 AI Chat**: Ask questions about your documents or websites
        - **💾 Chat History**: Persistent chat sessions
        - **🔧 Smart Processing**: Advanced text chunking and embeddings
        """)
        return

    # Initialize current chat ID
    if "current_chat_id" not in st.session_state:
        available_chats = get_available_chats(file_hash)
        st.session_state.current_chat_id = max(available_chats, default=0) + 1

    # Enhanced Sidebar
    with st.sidebar:
        st.header("💬 Chat Management")
        
        # Display current source info
        if source_name.startswith("PDF:"):
            st.info(f"📄 Document: {source_name[4:25]}...")
        else:
            st.info(f"🌐 Website: {source_name[8:35]}...")
        
        # New chat button
        if st.button("➕ New Chat", key="sidebar_new_chat_button"):
            available_chats = get_available_chats(file_hash)
            st.session_state.current_chat_id = max(available_chats, default=0) + 1
            st.session_state.chat_history = [AIMessage(content="Hi! I'm your document assistant. How can I help you today?")]
            if "chat_engine" in st.session_state:
                st.session_state.chat_engine.reset()
            st.rerun()

        # Chat selection
        available_chats = get_available_chats(file_hash)
        if available_chats:
            st.write("📑 Previous Chats:")
            for chat_id in sorted(available_chats, reverse=True)[:5]:
                if st.button(f"Chat {chat_id}", key=f"sidebar_chat_select_{chat_id}"):
                    st.session_state.current_chat_id = chat_id
                    st.session_state.chat_history = load_chat_history(chat_id, file_hash)
                    if "chat_engine" in st.session_state:
                        st.session_state.chat_engine.reset()
                    st.rerun()

        # Delete chat functionality
        if available_chats:
            st.write("🗑️ Delete Chats:")
            chat_to_delete = st.selectbox(
                "Select chat to delete:", 
                sorted(available_chats, reverse=True),
                key="sidebar_delete_chat_select"
            )
            if st.button("🗑️ Delete Selected", key="sidebar_delete_chat_button"):
                try:
                    with get_db_context() as db:
                        document = get_document_by_hash(db, file_hash)
                        if document:
                            chat = db.query(Chat).filter(
                                Chat.document_id == document.id,
                                Chat.chat_id == chat_to_delete
                            ).first()
                            if chat:
                                db.query(Message).filter(Message.chat_id == chat.id).delete()
                                db.delete(chat)
                                db.commit()
                                
                                if chat_to_delete == st.session_state.current_chat_id:
                                    st.session_state.current_chat_id = max(get_available_chats(file_hash), default=0) + 1
                                    st.session_state.chat_history = [AIMessage(content="Hi! I'm your document assistant. How can I help you today?")]
                                st.rerun()
                except Exception as e:
                    st.error(f"Failed to delete chat: {str(e)}")

        st.header("🔧 System Controls")
        # Force complete reset button
        if st.button("🔄 Force Complete Reset", key="force_reset_sidebar_button"):
            st.cache_data.clear()
            st.cache_resource.clear()
            
            old_persist_dir = f"vector_stores"
            if os.path.exists(old_persist_dir):
                try:
                    shutil.rmtree(old_persist_dir)
                except Exception as e:
                    st.warning(f"Could not delete vector stores: {e}")
            
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            
            st.success("🔄 Complete system reset done!")
            st.rerun()
        
        st.header("📞 Contact Information") 
        st.write(f"📧 **Email**: {COMPANY_EMAIL}")
        st.write(f"📱 **Phone**: {COMPANY_PHONE}")
        
        st.header("💡 Current Session")
        st.info(f"Chat ID: {st.session_state.current_chat_id}")
        st.info(f"Source Hash: {file_hash[:8]}...")
        
        st.header("🔧 System Status")
        with get_db_context() as db:
            document = get_document_by_hash(db, file_hash)
            if document:
                if document.content_type == 'website':
                    st.success("✅ Website data loaded")
                else:
                    st.success("✅ PDF loaded in database")
                st.info(f"📂 Source: {document.filename}")
            else:
                st.error("❌ Source not found")
        
        persist_dir = f"vector_stores/store_{file_hash}"
        if os.path.exists(os.path.join(persist_dir, "index_store.json")):
            st.success("✅ Vector store ready")
        elif "chat_engine" in st.session_state:
            st.success("✅ Chat system active")
        else:
            st.warning("⚠️ Initializing system...")

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = load_chat_history(st.session_state.current_chat_id, file_hash)

    # Initialize chat engine with force rebuild
    if "chat_engine" not in st.session_state:
        try:
            with st.spinner("🚀 Building fresh system for current source..."):
                force_rebuild = st.session_state.get('force_rebuild', True)
                index, bm25_retriever = load_or_build_index(file_hash, force_rebuild)
                
                chat_engine = create_chat_engine(index)
                
                if chat_engine is None:
                    st.error("Failed to create chat engine.")
                    st.stop()
                
                st.session_state.chat_engine = chat_engine
                st.session_state.force_rebuild = False
                st.success("✅ Fresh chat system ready!")
                
        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}")
            st.error(f"Initialization failed: {str(e)}")
            st.stop()

    # Display chat history
    for message in st.session_state.chat_history:
        role = "assistant" if isinstance(message, AIMessage) else "user"
        with st.chat_message(role):
            st.write(message.content)

    # User input with unique key
    user_query = st.chat_input("Ask me anything about the uploaded document or crawled website...", key="main_chat_input")
    if user_query and user_query.strip():
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        
        with st.chat_message("user"):
            st.write(user_query)
        
        with st.chat_message("assistant"):
            try:
                logger.info(f"🎯 PROCESSING QUERY: '{user_query}'")
                
                with st.spinner("🧠 Analyzing content..."):
                    response = st.session_state.chat_engine.chat(user_query)
                    response_text = response.response
                    logger.info(f"🎉 Generated response ({len(response_text)} chars)")
                
                st.write(response_text)
                st.session_state.chat_history.append(AIMessage(content=response_text))
                save_chat_history(st.session_state.current_chat_id, st.session_state.chat_history, file_hash)
                
            except Exception as e:
                logger.error(f"❌ Error: {str(e)}")
                error_message = "I apologize, but I encountered an error. Please try rephrasing your question."
                st.write(error_message)
                st.session_state.chat_history.append(AIMessage(content=error_message))

if __name__ == "__main__":
    main()
