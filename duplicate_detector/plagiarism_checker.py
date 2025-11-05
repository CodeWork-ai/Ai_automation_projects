import os
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from api_clients import get_groq_client, get_firecrawl_client, get_google_search_service

# This is a one-time download for the sentence tokenizer.
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK's 'punkt' tokenizer for the first time...")
    nltk.download('punkt')

def chunk_text(text):
    """Breaks the input text into a list of sentences."""
    return nltk.sent_tokenize(text)

def search_web_for_chunk(chunk, search_service, cse_id):
    """Searches a single text chunk on the web and returns top URLs."""
    try:
        res = search_service.cse().list(q=chunk, cx=cse_id, num=3).execute()
        return [item['link'] for item in res.get('items', [])]
    except Exception as e:
        print(f"An error occurred during web search: {e}")
        return []

def scrape_url_content(url, firecrawl_client):
    """Scrapes the main content of a URL."""
    try:
        scraped_data = firecrawl_client.scrape(
            url,
            only_main_content=True
        )
        if hasattr(scraped_data, 'markdown'):
            return scraped_data.markdown
        return None
    except Exception as e:
        print(f"--- FAILED TO SCRAPE: {url} ---")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Details: {e}")
        return None




def crawl_website(url, firecrawl_client):
    """Crawls a website and extracts markdown content from all pages."""
    print(f"Crawling {url}...")
    try:
        # Corrected: Pass pageOptions as a direct keyword argument
        crawl_results = firecrawl_client.crawl_url(
            url, 
            page_options={'onlyMainContent': True}
        )
        return {item['url']: item['markdown'] for item in crawl_results if item.get('markdown')}
    except Exception as e:
        print(f"An error occurred during crawling: {e}")
        return {}


def read_local_directory(path):
    """Reads all .txt and .md files from a local directory."""
    print(f"Reading files from {path}...")
    content_map = {}
    try:
        for root, _, files in os.walk(path):
            for file_name in files:
                if file_name.endswith(('.txt', '.md')):
                    file_path = os.path.join(root, file_name)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content_map[file_path] = f.read()
    except Exception as e:
        print(f"An error occurred reading local files: {e}")
    return content_map

def find_internal_duplicates(content_map, similarity_threshold=0.8):
    """Analyzes a map of documents to find internal similarities."""
    if len(content_map) < 2:
        return {}
    file_names = list(content_map.keys())
    documents = list(content_map.values())
    vectorizer = TfidfVectorizer(stop_words='english').fit_transform(documents)
    cosine_sim_matrix = cosine_similarity(vectorizer)
    duplicates = {}
    for i in range(len(cosine_sim_matrix)):
        for j in range(i + 1, len(cosine_sim_matrix)):
            if cosine_sim_matrix[i][j] > similarity_threshold:
                file1, file2 = file_names[i], file_names[j]
                if file1 not in duplicates: duplicates[file1] = []
                duplicates[file1].append({'similar_to': file2, 'score': cosine_sim_matrix[i][j]})
    return duplicates

def compare_with_llm(original_text, scraped_text, source_url, groq_client):
    """Uses Groq LLM to compare original text with scraped text."""
    # ... (function content remains the same)
    prompt = f"""...""" # Your prompt here
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"An error occurred during LLM analysis: {e}"

def generate_optimization_report(content, plagiarism_results, groq_client):
    """Generates a final report with a uniqueness score and optimization tips."""
    # ... (function content remains the same)
    highest_similarity = 85 # Placeholder
    uniqueness_score = 100 - highest_similarity
    prompt = f"""...""" # Your prompt here
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"An error occurred during report generation: {e}"
