import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

def setup_vector_database():
    if os.path.exists("chroma_db"):
        print("ChromaDB already exists. Skipping.")
        return
    print("Setting up ChromaDB...")
    loader = DirectoryLoader("knowledge_base", glob="*.txt")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma.from_documents(texts, embeddings, persist_directory="chroma_db")
    print("✅ ChromaDB setup complete!")

if __name__ == "__main__":
    setup_vector_database()