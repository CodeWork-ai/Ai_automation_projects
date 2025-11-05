import os
from groq import Groq
from googleapiclient.discovery import build
from dotenv import load_dotenv
from firecrawl import Firecrawl # This is the correct class to import

# Load environment variables from the .env file
load_dotenv()

def get_groq_client():
    """Initializes and returns the Groq API client."""
    try:
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        return client
    except Exception as e:
        print(f"Error initializing Groq client: {e}")
        return None

# In api_clients.py



def get_firecrawl_client():
    """Initializes and returns the Firecrawl client."""
    try:
        # The class name is Firecrawl
        client = Firecrawl(api_key=os.environ.get("FIRECRAWL_API_KEY"))
        return client
    except Exception as e:
        print(f"Error initializing FireCrawl client: {e}")
        return None



def get_google_search_service():
    """Initializes and returns the Google Custom Search service."""
    try:
        service = build("customsearch", "v1", developerKey=os.environ.get("GOOGLE_API_KEY"))
        return service
    except Exception as e:
        print(f"Error initializing Google Search service: {e}")
        return None