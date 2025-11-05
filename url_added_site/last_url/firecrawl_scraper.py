import requests
from config import FIRECRAWL_API_KEY
import re

def fetch_url_with_firecrawl(url: str):
    if not FIRECRAWL_API_KEY:
        raise ValueError("FIRECRAWL_API_KEY is not set.")
    
    endpoint = "https://api.firecrawl.dev/v1/scrape"
    headers = {
        "Authorization": f"Bearer {FIRECRAWL_API_KEY}",
        "Content-Type": "application/json"
    }
    # Firecrawl expects a JSON payload, not params in the URL
    payload = {"url": url}
    try:
        resp = requests.post(endpoint, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        print(f"Request Error fetching {url}: {e}")
        return None

def search_amazon_for_product_urls(params: dict, domain: str = "com", limit: int = 3) -> list[str]:
    """
    Constructs a filtered Amazon search URL and extracts the top product URLs.
    """
    query = params.get("keywords", "")
    if params.get("brand"):
        query += f' {params.get("brand")}'
    
    base_url = f"https://www.amazon.{domain}/s?k={query.replace(' ', '+')}"
    
    min_price = params.get("min_price")
    max_price = params.get("max_price")
    
    if min_price is not None and max_price is not None:
        base_url += f"&rh=p_36%3A{int(min_price)}00-{int(max_price)}00"
    elif max_price is not None:
         base_url += f"&rh=p_36%3A-{int(max_price)}00"
    
    print(f"Searching Amazon with filtered URL: {base_url}")

    scraped_data = fetch_url_with_firecrawl(base_url)
    if not scraped_data or not scraped_data.get("data") or not scraped_data["data"].get("markdown"):
        return []

    content = scraped_data["data"]["markdown"]
    
    url_pattern = r'https://www\.amazon\.' + re.escape(domain) + r'/[^/]+/dp/[A-Z0-9]{10}'
    found_urls = re.findall(url_pattern, content)
    
    unique_urls = list(dict.fromkeys(found_urls)) # Remove duplicates
    print(f"Found {len(unique_urls)} unique product URLs.")
    return unique_urls[:limit]