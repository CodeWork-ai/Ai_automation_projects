import json
import re
import time
from database import get_db
from firecrawl_scraper import fetch_url_with_firecrawl
from llm_client import LLMClient
from sqlite_utils.db import NotFoundError

class BaseAgent:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def answer_question_from_context(self, question: str, url: str, history: list) -> str:
        db = get_db()
        
        # Handle both original URLs and comprehensive cache keys
        cache_key = self._get_cache_key(url) # Use a helper to determine cache key
        
        try:
            cached_data = db[self.table_name].get(cache_key)
            content = cached_data["data"]
            title = cached_data.get("name", "this entity")
        except (NotFoundError, KeyError):
            return "Error: Could not find the data for this entity. Please try loading the URL again."

        history_str = "\n".join([f"User: {q}\nAssistant: {a}" for q, a in history])

        # This is the NEW prompt for your BaseAgent class
        prompt = f"""
        You are an expert analyst and assistant answering questions about '{title}'.

        The context below contains comprehensive information from multiple pages of the entity's profile, clearly marked by section headers (e.g., "=== FINANCIALS SECTION ==="). Use ALL relevant information from ANY section to provide a complete and accurate answer.

        **Context:**
        ---
        {content[:15000]}
        ---

        **Conversation History:**
        ---
        {history_str}
        ---

        **User Question:** {question}

        ---
        **RESPONSE INSTRUCTIONS:**
        Your answer MUST be structured using Markdown. Follow these rules precisely:
        1.  **Use Headings:** Organize your answer into sections with clear Markdown headings (e.g., `### Key Financials`, `### Recent News`).
        2.  **Use Bullet Points:** List key data points like funding rounds, investors, or personnel using bullet points (`-`).
        3.  **Be Specific:** Provide concrete numbers, dates, and names found in the context.
        4.  **Synthesize:** If the answer requires information from multiple sections, bring it together into a coherent response under the appropriate heading.
        5.  **If a question is broad** (e.g., "Tell me about this company"), structure your answer with logical headings like `### Overview`, `### Financial Performance`, and `### Key Statistics`.
        6.  **If information is not available**, clearly state that. Do not make up information.

        **Formatted Answer:**
        """


        return self.llm.ask_llm(prompt)

    def get_cached_data(self, url: str) -> str | None:
        """Retrieves raw data from the cache for a given URL."""
        db = get_db()
        cache_key = self._get_cache_key(url)
        try:
            cached_data = db[self.table_name].get(cache_key)
            return cached_data.get("data")
        except (NotFoundError, KeyError):
            return None
    
    def _get_cache_key(self, url: str) -> str:
        """Helper method to be implemented by subclasses for generating a unique cache key."""
        raise NotImplementedError

    def load_and_cache_data(self, url: str) -> str | None:
        """Main method to be implemented by subclasses to handle data loading."""
        raise NotImplementedError


class CrunchbaseAgent(BaseAgent):
    def __init__(self, llm: LLMClient):
        super().__init__(llm)
        self.table_name = "crunchbase"

    def _get_cache_key(self, url: str) -> str:
        org_name = url.split("organization/")[-1].split("/")[0]
        base_url = f"https://www.crunchbase.com/organization/{org_name}"
        return f"{base_url}/comprehensive"
    
    def get_standard_crunchbase_sections(self):
        """Standard sections available on most Crunchbase organization pages"""
        return [
            "",  # Overview (main page)
            "/predictions_and_insights",
            "/growth_outlook", 
            "/financials",
            "/financial_details",
            "/people",
            "/profiles_and_contacts",
            "/news",
            "/news_and_analysis", 
            "/technology",
            "/tech_details",
            "/lists_featuring_this_company",
            "/frequently_asked_questions"
        ]

    def load_and_cache_data(self, url: str) -> str | None:
        """Enhanced method that automatically fetches comprehensive data"""
        db = get_db()
        org_name = url.split('/organization/')[-1].split('/')[0]
        base_org_url = f"https://www.crunchbase.com/organization/{org_name}"
        cache_key = self._get_cache_key(url)

        print(f"Crunchbase Agent: Loading comprehensive data for {org_name}")
        
        try:
            cached_data = db[self.table_name].get(cache_key)
            if cached_data and cached_data.get("data"):
                print("Using cached comprehensive Crunchbase data.")
                return cached_data["data"]
        except NotFoundError:
            pass

        sections = self.get_standard_crunchbase_sections()
        all_content = []
        successful_pages = 0
        
        for section in sections:
            page_url = base_org_url + section
            try:
                print(f"Fetching: {page_url}")
                data = fetch_url_with_firecrawl(page_url)
                if data and data.get("data") and data["data"].get("markdown"):
                    content = data["data"].get("markdown", "")
                    if content.strip():
                        section_name = section.replace("/", " ").strip() or "overview"
                        all_content.append(f"=== {section_name.upper()} SECTION ===\n{content}\n")
                        successful_pages += 1
                        print(f"✓ Fetched data from: {section_name}")
                    else:
                        print(f"✗ Empty content from: {page_url}")
                else:
                    print(f"✗ Failed to fetch: {page_url}")
                
                time.sleep(1.5)
                
            except Exception as e:
                print(f"✗ Error fetching {page_url}: {e}")
                continue

        if successful_pages == 0:
            return None

        comprehensive_content = "\n".join(all_content)
        
        metadata = {"title": f"{org_name} - Comprehensive Data", "description": f"Complete data from {successful_pages} sections"}
        db[self.table_name].upsert({
            "url": cache_key, 
            "name": metadata["title"], 
            "description": metadata["description"], 
            "data": comprehensive_content
        }, pk="url")
        
        print(f"Fetched and cached comprehensive data from {successful_pages} sections.")
        return comprehensive_content


class YahooFinanceAgent(BaseAgent):
    def __init__(self, llm: LLMClient):
        super().__init__(llm)
        self.table_name = "yahoo_finance"

    def _get_cache_key(self, url: str) -> str:
        ticker = url.split("quote/")[-1].split("/")[0]
        base_url = f"https://finance.yahoo.com/quote/{ticker}"
        return f"{base_url}/comprehensive"
    
    def get_standard_yahoo_finance_sections(self):
        """Standard sections for a stock ticker on Yahoo Finance."""
        return [
            "", # Summary
            "/key-statistics",
            "/financials",
            "/analysis",
            "/holders",
        ]
    
    def load_and_cache_data(self, url: str) -> str | None:
        """Fetches comprehensive data for a stock ticker from Yahoo Finance."""
        db = get_db()
        ticker = url.split('quote/')[-1].split('/')[0]
        base_quote_url = f"https://finance.yahoo.com/quote/{ticker}"
        cache_key = self._get_cache_key(url)

        print(f"Yahoo Finance Agent: Loading comprehensive data for {ticker}")
        
        try:
            cached_data = db[self.table_name].get(cache_key)
            if cached_data and cached_data.get("data"):
                print("Using cached comprehensive Yahoo Finance data.")
                return cached_data["data"]
        except NotFoundError:
            pass

        sections = self.get_standard_yahoo_finance_sections()
        all_content = []
        successful_pages = 0
        
        for section in sections:
            page_url = base_quote_url + section
            try:
                print(f"Fetching: {page_url}")
                data = fetch_url_with_firecrawl(page_url)
                if data and data.get("data") and data["data"].get("markdown"):
                    content = data["data"].get("markdown", "")
                    if content.strip():
                        section_name = section.replace("/", " ").strip() or "summary"
                        all_content.append(f"=== {section_name.upper()} SECTION ===\n{content}\n")
                        successful_pages += 1
                        print(f"✓ Fetched data from: {section_name}")
                    else:
                        print(f"✗ Empty content from: {page_url}")
                else:
                    print(f"✗ Failed to fetch: {page_url}")
                
                time.sleep(1.5)
                
            except Exception as e:
                print(f"✗ Error fetching {page_url}: {e}")
                continue

        if successful_pages == 0:
            return None

        comprehensive_content = "\n".join(all_content)
        
        metadata = {"title": f"{ticker} - Comprehensive Data", "description": f"Complete data from {successful_pages} sections"}
        db[self.table_name].upsert({
            "url": cache_key, 
            "name": metadata["title"], 
            "description": metadata["description"], 
            "data": comprehensive_content
        }, pk="url")
        
        print(f"Fetched and cached comprehensive data from {successful_pages} sections.")
        return comprehensive_content


class AmazonAgent(BaseAgent):
    def __init__(self, llm: LLMClient):
        super().__init__(llm)
        self.table_name = "amazon_products"

    def _get_cache_key(self, url: str) -> str:
        # For Amazon, the URL itself is a good unique key
        return url

    def load_and_cache_data(self, url: str) -> str | None:
        db = get_db()
        print(f"Amazon Agent: Loading data for URL {url}")

        try:
            cached_data = db[self.table_name].get(url)
            if cached_data and cached_data.get("data"):
                print("Using cached Amazon data.")
                return cached_data["data"]
        except NotFoundError:
            pass

        data = fetch_url_with_firecrawl(url)
        if not data or not data.get("data") or not data.get("markdown"):
            return None

        content = data["data"].get("markdown", "")
        db[self.table_name].upsert({"url": url, "data": content}, pk="url")
        print("Fetched and cached new Amazon data.")
        return content

    def get_product_details(self, url: str) -> dict | None:
        content = self.load_and_cache_data(url)
        if not content:
            return None

        prompt = f"""
Analyze the markdown from an Amazon product page below. Extract details and return a valid JSON object.

CRITICAL INSTRUCTIONS:
1. IGNORE any information related to "Asurion", "Protection Plans", or "Warranties".
2. "key_features" MUST come from the "About this item" section.
3. Create a "review_summary" by summarizing the general customer sentiment.

JSON Structure:
{{
"product_name": "...",
"price": "...",
"rating": "...",
"key_features": ["...", "..."],
"review_summary": "..."
}}

Page Content:
---
{content[:24000]}
---

JSON Output:
"""

        json_string = self.llm.extract_product_details_json(prompt)
        try:
            sanitized_json_string = json_string.replace(r'\|', '|')
            details = json.loads(sanitized_json_string)
            details['product_url'] = url
            return details
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error parsing product details for {url}: {e}")
            return None