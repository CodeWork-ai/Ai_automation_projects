from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
import os



# Use the new import path for declarative_base
Base = declarative_base()

class ConversationLog(Base):
    __tablename__ = 'conversations'
    
    id = Column(Integer, primary_key=True)
    conversation_id = Column(String, unique=True, nullable=False)
    conversation_history = Column(Text, nullable=False)  # JSON string
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
# Default database URL
DEFAULT_DB_URL = os.getenv("DB_URL", "sqlite:///weather_agent.db")

# Create engine
engine = create_engine(DEFAULT_DB_URL, future=True)

# Create all tables
Base.metadata.create_all(engine)

# Create a session factory
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)

# Create a global Session instance
Session = SessionLocal

def init_db(db_url=None):
    """Initialize database with custom URL"""
    if db_url:
        engine = create_engine(db_url, future=True)
        Base.metadata.create_all(engine)
        return sessionmaker(bind=engine, expire_on_commit=False)
    return SessionLocal