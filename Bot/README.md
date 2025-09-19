# Document AI Assistant

## Overview
Document AI Assistant is an intelligent chatbot application that allows users to upload PDF and DOCX documents and interact with their content through a conversational interface. The application uses advanced RAG (Retrieval-Augmented Generation) techniques to provide accurate, context-aware responses based on the document content.

## Features
- **Document Processing**: Upload and process PDF and DOCX files
- **Intelligent Retrieval**: Hybrid search combining vector embeddings and BM25 for optimal results
- **Conversational Interface**: Chat with your documents using natural language
- **Context Memory**: The assistant remembers conversation history for more coherent interactions
- **Web Interface**: User-friendly interface with real-time responses
- **Database Storage**: Persistent storage of documents and chat history
- **Error Handling**: Robust error handling and logging

## Technology Stack

### Backend
- **FastAPI**: Modern, high-performance web framework for building APIs
- **SQLAlchemy**: SQL toolkit and Object-Relational Mapping (ORM) library
- **PostgreSQL**: Database with pgvector extension for vector similarity search
- **LlamaIndex**: Framework for building LLM-powered applications
- **LangChain**: Framework for developing applications powered by language models
- **OpenAI**: Integration with OpenAI's language models

### Frontend
- **HTML/CSS/JavaScript**: Custom web interface
- **WebSockets**: Real-time communication between client and server
- **Font Awesome**: Icons for improved UI

## Installation

### Prerequisites
- Python 3.8 or higher
- PostgreSQL with pgvector extension
- OpenAI API key
- Firecrawl API key (for web crawling functionality)

### Setup

1. Clone the repository

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   # On Windows
   .venv\Scripts\activate
   # On macOS/Linux
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up PostgreSQL database:
   - Create a database named `faqchatbot`
   - Enable pgvector extension

5. Create a `.env` file in the project root with the following variables:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   FIRECRAWL_API_KEY=your_firecrawl_api_key_here
   ```

6. Initialize the database:
   ```bash
   alembic upgrade head
   ```

## Usage

### Running the Application

#### Streamlit Interface
```bash
streamlit run app.py
```

#### FastAPI Backend
```bash
uvicorn main:app --reload
```

### Using the Application

1. **Upload a Document**:
   - Click on the "Choose PDF File" button in the sidebar
   - Select a PDF or DOCX file from your computer
   - Wait for the document to be processed

2. **Chat with the Document**:
   - Once the document is processed, the chat input will be enabled
   - Type your questions about the document in the chat input
   - The AI assistant will respond with relevant information from the document

3. **Manage Chats**:
   - Use the "New Chat" button to start a fresh conversation
   - Previous chats are saved and can be accessed from the sidebar

## Project Structure

- `app.py`: Streamlit application for the web interface
- `main.py`: FastAPI application for the backend API
- `database.py`: Database models and connection setup
- `document_processor.py`: Document text extraction utilities
- `index.html`, `style.css`, `script.js`: Frontend web interface files
- `migrations/`: Alembic database migration files
- `vector_stores/`: Directory for storing vector indices

## License

This project is proprietary software. All rights reserved.

## Contact

For support or inquiries, contact:
- Email: sales@codework.ai
- Phone: +91 75989 81500