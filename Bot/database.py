from sqlalchemy import create_engine, Column, Integer, String, LargeBinary, DateTime, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.dialects.postgresql import ARRAY, REAL
from datetime import datetime
import pgvector.sqlalchemy

# Database Configuration
from sqlalchemy.engine.url import URL

def get_sync_url():
    return URL.create(
        drivername="postgresql",
        username="postgres",
        password="root",
        host="localhost",
        port=5432,
        database="faqchatbot"
    )

def get_async_url():
    return URL.create(
        drivername="postgresql+asyncpg",
        username="postgres",
        password="root",
        host="localhost",
        port=5432,
        database="faqchatbot"
    )

DATABASE_URL = str(get_sync_url())
ASYNC_DATABASE_URL = str(get_async_url())

# Create SQLAlchemy engine
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Enable pgvector extension
def enable_pgvector(engine):
    with engine.connect() as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS pgvector")
        conn.commit()

class DocumentChunk(Base):
    __tablename__ = "document_chunks"
    
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"))
    content = Column(Text)
    embedding = Column(ARRAY(REAL))  # Store embeddings as array of floats initially
    chunk_number = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class Document(Base):
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    file_hash = Column(String, unique=True, index=True)
    content_type = Column(String)  # 'pdf' or 'docx'
    file_data = Column(LargeBinary)
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    chats = relationship("Chat", back_populates="document")
    chunks = relationship("DocumentChunk", backref="document")

class Chat(Base):
    __tablename__ = "chats"
    
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"))
    chat_id = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    document = relationship("Document", back_populates="chats")
    messages = relationship("Message", back_populates="chat")

class Message(Base):
    __tablename__ = "messages"
    
    id = Column(Integer, primary_key=True, index=True)
    chat_id = Column(Integer, ForeignKey("chats.id"))
    role = Column(String)  # 'user' or 'assistant'
    content = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)
    chat = relationship("Chat", back_populates="messages")

def init_db():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
