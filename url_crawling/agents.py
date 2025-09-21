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
        cache_key = url
        if '/organization/' in url:
            org_name = url.split('/organization/')[-1].split('/')[0]
            base_url = f"https://www.crunchbase.com/organization/{org_name}"
            cache_key = f"{base_url}_comprehensive"
        
        try:
            cached_data = db[self.table_name].get(cache_key)
            content = cached_data["data"]
            title = cached_data.get("name", "this organization")
        except (NotFoundError, KeyError):
            return "Error: Could not find the data for this organization. Please try loading the URL again."

        history_str = "\n".join([f"User: {q}\nAssistant: {a}" for q, a in history])

        # This is the NEW prompt for your BaseAgent class
        prompt = f"""
        You are a business analyst and expert assistant answering questions about the Crunchbase organization '{title}'.

        The context below contains comprehensive information from multiple pages of the organization's profile, clearly marked by section headers (e.g., "=== FINANCIALS SECTION ==="). Use ALL relevant information from ANY section to provide a complete and accurate answer.

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
        4.  **Synthesize:** If the answer requires information from multiple sections (e.g., combining `FINANCIALS` and `NEWS`), bring it together into a coherent response under the appropriate heading.
        5.  **If a question is broad** (e.g., "Tell me about this company"), structure your answer with logical headings like `### Overview`, `### Funding`, and `### Key People`.
        6.  **If information is not available**, clearly state that. Do not make up information.

        **Formatted Answer:**
        """


        return self.llm.ask_gemini(prompt)

class CrunchbaseAgent(BaseAgent):
    def __init__(self, llm: LLMClient):
        super().__init__(llm)
        self.table_name = "crunchbase"
        # Remove the old hardcoded sub_pages list

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
        # Extract base organization URL
        if '/organization/' in url:
            org_name = url.split('/organization/')[-1].split('/')[0]
            base_url = f"https://www.crunchbase.com/organization/{org_name}"
            
            # Use comprehensive data fetching
            return self.load_and_cache_comprehensive_data(base_url)
        
        # Fallback to original method for non-organization URLs
        return self.original_load_and_cache_data(url)

    def load_and_cache_comprehensive_data(self, base_url: str) -> str | None:
        """Fetch data from all standard Crunchbase sections"""
        db = get_db()
        
        # Extract organization name from URL
        org_name = base_url.split('/organization/')[-1].split('/')[0]
        base_org_url = f"https://www.crunchbase.com/organization/{org_name}"
        
        print(f"Crunchbase Agent: Loading comprehensive data for {org_name}")
        
        # Check if we have cached comprehensive data
        cache_key = f"{base_org_url}_comprehensive"
        try:
            cached_data = db[self.table_name].get(cache_key)
            if cached_data and cached_data.get("data"):
                print("Using cached comprehensive Crunchbase data.")
                return cached_data["data"]
        except NotFoundError:
            pass

        # Get all standard sections
        sections = self.get_standard_crunchbase_sections()
        
        all_content = []
        successful_pages = 0
        
        for section in sections:
            url = base_org_url + section
            try:
                print(f"Fetching: {url}")
                data = fetch_url_with_firecrawl(url)
                if data and data.get("data") and data["data"].get("markdown"):
                    content = data["data"].get("markdown", "")
                    if content.strip():  # Only add non-empty content
                        section_name = section.replace("/", " ").strip() or "overview"
                        all_content.append(f"=== {section_name.upper()} SECTION ===\n{content}\n")
                        successful_pages += 1
                        print(f"✓ Fetched data from: {section_name}")
                    else:
                        print(f"✗ Empty content from: {url}")
                else:
                    print(f"✗ Failed to fetch: {url}")
                
                # Add delay to avoid rate limiting
                time.sleep(1.5)
                
            except Exception as e:
                print(f"✗ Error fetching {url}: {e}")
                continue

        if successful_pages == 0:
            return None

        # Combine all content
        comprehensive_content = "\n".join(all_content)
        
        # Cache the comprehensive data
        metadata = {"title": f"{org_name} - Comprehensive Data", "description": f"Complete data from {successful_pages} sections"}
        db[self.table_name].upsert({
            "url": cache_key, 
            "name": metadata["title"], 
            "description": metadata["description"], 
            "data": comprehensive_content
        }, pk="url")
        
        print(f"Fetched and cached comprehensive data from {successful_pages} sections.")
        return comprehensive_content


    def original_load_and_cache_data(self, url: str) -> str | None:
        """Original single-page data loading method"""
        db = get_db()
        print(f"Crunchbase Agent: Loading data for URL {url}")

        try:
            cached_data = db[self.table_name].get(url)
            if cached_data and cached_data.get("data"):
                print("Using cached Crunchbase data.")
                return cached_data["data"]
        except NotFoundError:
            pass

        data = fetch_url_with_firecrawl(url)
        if not data or not data.get("data") or not data["data"].get("markdown"):
            return None

        content = data["data"].get("markdown", "")
        name = data["data"].get("metadata", {}).get("title", "Unknown Company")
        description = data["data"].get("metadata", {}).get("description", "")

        db[self.table_name].upsert({
            "url": url, "name": name, "description": description, "data": content
        }, pk="url")

        print("Fetched and cached new Crunchbase data.")
        return content

class AmazonAgent(BaseAgent):
    def __init__(self, llm: LLMClient):
        super().__init__(llm)
        self.table_name = "amazon_products"

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
        if not data or not data.get("data") or not data["data"].get("markdown"):
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

        json_string = self.llm.extract_json(prompt)
        try:
            sanitized_json_string = json_string.replace(r'\|', '|')
            details = json.loads(sanitized_json_string)
            details['url'] = url
            return details
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error parsing product details for {url}: {e}")
            return None
