from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import os
import json
import hashlib
import tempfile
import logging
from typing import Dict, List
import asyncio

# Import your existing RAG functions
from database import SessionLocal, init_db, Document, Chat, Message
from sqlalchemy.orm import Session
from contextlib import contextmanager

# LlamaIndex imports from your existing code
from llama_index.core import VectorStoreIndex, StorageContext, Settings, Document as LlamaDocument
from llama_index.core.schema import TextNode
from llama_index.readers.file import PyMuPDFReader as PDFReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.retrievers.bm25 import BM25Retriever

from dotenv import load_dotenv

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Initialize FastAPI app
app = FastAPI(title="Document AI Assistant API", version="1.0.0")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (HTML, CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize database
init_db()

# Global variables
chat_engines: Dict[str, ContextChatEngine] = {}
connected_clients: List[WebSocket] = []

# Database context
@contextmanager
def get_db_context():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Pydantic models
class ChatMessage(BaseModel):
    message: str
    file_hash: str
    chat_id: int = 1

class ChatResponse(BaseModel):
    response: str
    chat_id: int

class UploadResponse(BaseModel):
    filename: str
    file_hash: str
    status: str

# Helper functions from your existing code
def get_document_by_hash(db: Session, file_hash: str):
    return db.query(Document).filter(Document.file_hash == file_hash).first()

def save_uploaded_file(db: Session, file_content: bytes, filename: str, file_hash: str):
    db_document = Document(
        filename=filename,
        file_hash=file_hash,
        content_type='pdf',
        file_data=file_content
    )
    db.add(db_document)
    db.commit()
    db.refresh(db_document)
    return db_document

def preprocess_pdf_text(text):
    import re
    text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = re.sub(r'^\s*[•·\-\*]\s*', '- ', text, flags=re.MULTILINE)
    return text.strip()

async def process_document(file_hash: str, file_content: bytes, filename: str):
    """Process PDF and create chat engine"""
    try:
        # Configure embeddings
        # Initialize OpenAIEmbedding without the 'proxies' parameter
        # The 'proxies' parameter is not supported by LlamaIndex's OpenAIEmbedding class
        # Removed proxies parameter to fix "init() got an unexpected keyword argument 'proxies'" error
        Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
        
        # Save to database
        with get_db_context() as db:
            existing_doc = get_document_by_hash(db, file_hash)
            if not existing_doc:
                save_uploaded_file(db, file_content, filename, file_hash)
        
        # Process PDF
        temp_pdf = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
        try:
            temp_pdf.write(file_content)
            temp_pdf.flush()
            temp_pdf.close()
            
            loader = PDFReader()
            raw_docs = loader.load_data(temp_pdf.name)
        finally:
            os.unlink(temp_pdf.name)
        
        if not raw_docs:
            raise ValueError("No content found in PDF")
        
        # Process text
        combined_text = "\n\n".join([doc.text for doc in raw_docs])
        combined_text = preprocess_pdf_text(combined_text)
        
        # Create chunks
        splitter = SentenceSplitter(chunk_size=300, chunk_overlap=60)
        enhanced_doc = LlamaDocument(
            text=combined_text,
            metadata={"filename": filename, "file_hash": file_hash}
        )
        nodes = splitter.get_nodes_from_documents([enhanced_doc])
        
        # Create vector index
        vector_store = SimpleVectorStore()
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex(nodes=nodes, storage_context=storage_context)
        
        # Create chat engine
        memory = ChatMemoryBuffer.from_defaults(token_limit=8000)
        chat_engine = ContextChatEngine.from_defaults(
            retriever=index.as_retriever(similarity_top_k=8),
            memory=memory,
            system_prompt=(
                "You are an AI assistant that helps users understand documents. "
                "Use ONLY the provided context to give detailed, accurate responses. "
                "Never make up information that is not in the context. "
                "Be helpful and thorough in your responses."
            ),
        )
        
        # Store chat engine
        chat_engines[file_hash] = chat_engine
        
        logger.info(f"Successfully processed document: {filename}")
        return True
        
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        raise e

# Routes
@app.get("/", response_class=HTMLResponse)
async def get_homepage():
    """Serve the main chat interface"""
    with open("static/index.html", "r") as f:
        return HTMLResponse(content=f.read())

@app.post("/api/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """Upload and process PDF file"""
    try:
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        
        # Read file content
        content = await file.read()
        file_hash = hashlib.md5(content).hexdigest()
        
        # Check if already processed
        if file_hash in chat_engines:
            return UploadResponse(
                filename=file.filename,
                file_hash=file_hash,
                status="already_processed"
            )
        
        # Process document
        await process_document(file_hash, content, file.filename)
        
        return UploadResponse(
            filename=file.filename,
            file_hash=file_hash,
            status="success"
        )
        
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat", response_model=ChatResponse)
async def chat_message(chat_data: ChatMessage):
    """Process chat message and return response"""
    try:
        file_hash = chat_data.file_hash
        message = chat_data.message
        chat_id = chat_data.chat_id
        
        # Check if chat engine exists
        if file_hash not in chat_engines:
            raise HTTPException(status_code=404, detail="Document not found or not processed")
        
        chat_engine = chat_engines[file_hash]
        
        # Generate response
        response = chat_engine.chat(message)
        
        return ChatResponse(
            response=response.response,
            chat_id=chat_id
        )
        
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """WebSocket endpoint for real-time chat"""
    await websocket.accept()
    connected_clients.append(websocket)
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            file_hash = message_data.get("file_hash")
            message = message_data.get("message")
            
            if file_hash and file_hash in chat_engines:
                # Generate response
                chat_engine = chat_engines[file_hash]
                response = chat_engine.chat(message)
                
                # Send response back
                await websocket.send_text(json.dumps({
                    "type": "response",
                    "message": response.response
                }))
            else:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Document not found or not processed"
                }))
                
    except WebSocketDisconnect:
        connected_clients.remove(websocket)

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "chat_engines": len(chat_engines)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
