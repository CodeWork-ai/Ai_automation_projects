# utils.py

import os
from dotenv import load_dotenv
import pypdf
from docx import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# Load environment variables from .env file
load_dotenv()

# Configure the Gemini API key
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env file")
os.environ["GOOGLE_API_KEY"] = api_key


def get_document_text(uploaded_files):
    """
    Extracts text from a list of uploaded files (PDFs or DOCXs).
    """
    text = ""
    for uploaded_file in uploaded_files:
        if uploaded_file.name.endswith('.pdf'):
            pdf_reader = pypdf.PdfReader(uploaded_file)
            for page in pdf_reader.pages:
                text += page.extract_text()
        elif uploaded_file.name.endswith('.docx'):
            doc = Document(uploaded_file)
            for para in doc.paragraphs:
                text += para.text + "\n"
    return text


def get_text_chunks(text):
    """
    Splits the text into smaller chunks for processing.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    """
    Creates and saves a vector store from text chunks using Google embeddings.
    """
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = Chroma.from_texts(text_chunks, embedding=embeddings)
    return vector_store


def get_conversational_chain():
    """
    Creates a conversational QA chain with a custom prompt for contract analysis.
    """
    prompt_template = """
    You are an expert legal AI assistant. Your task is to review a contract and provide a detailed analysis.
    Analyze the following contract document and provide the following:
    1.  **Summary of the Contract**: A concise overview of the contract's purpose, parties involved, and key terms.
    2.  **Identification of Risky Clauses**: Pinpoint clauses that are ambiguous, one-sided, unfavorable, or potentially problematic. For each clause, quote the clause text, explain the associated risks, and categorize the risk level (e.g., High, Medium, Low).
    3.  **Suggested Edits**: For each identified risky clause, propose specific alternative wording or edits to mitigate the risks and make the contract more balanced and fair.

    The user has provided the following contract context:
    \nContext:\n {context} \n

    Provide a well-structured, easy-to-read analysis based on this context.
    Analysis:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


def process_contract(uploaded_file):
    """
    Main function to orchestrate the contract review process.
    """
    # Step 1: Extract text from the document
    raw_text = get_document_text([uploaded_file])
    if not raw_text:
        return "Could not extract text from the document. Please check the file."

    # Step 2: Split text into chunks
    text_chunks = get_text_chunks(raw_text)

    # Step 3: Create a vector store
    vector_store = get_vector_store(text_chunks)

    # Step 4: Create the conversational chain
    chain = get_conversational_chain()

    # Step 5: Run the chain with the document chunks as context
    # Since load_qa_chain with "stuff" combines all docs, we can pass them directly.
    # We don't need a specific question, as the prompt guides the model.
    docs = vector_store.similarity_search(query="analyze the contract")
    response = chain({"input_documents": docs}, return_only_outputs=True)
    
    return response["output_text"]