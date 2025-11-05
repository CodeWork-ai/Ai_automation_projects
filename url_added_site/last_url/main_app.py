import re
from llm_client import LLMClient
from agents import AmazonAgent, CrunchbaseAgent, YahooFinanceAgent # Import new agent
from firecrawl_scraper import search_amazon_for_product_urls

class Assistant:
    def __init__(self):
        # The Assistant no longer manages the database connection directly.
        self.llm = LLMClient()
        # The agents no longer need the db object.
        self.amazon_agent = AmazonAgent(self.llm)
        self.crunchbase_agent = CrunchbaseAgent(self.llm)
        self.yahoo_finance_agent = YahooFinanceAgent(self.llm) # Instantiate new agent
        self.amazon_domain_tld = "in"

    def load_url_for_qa(self, url: str):
        print(f"Attempting to load comprehensive URL data for Q&A: {url}")
        
        content = None
        # Route to the correct agent based on URL
        if re.search(r'crunchbase\.com/organization/', url):
            content = self.crunchbase_agent.load_and_cache_data(url)
        elif re.search(r'finance\.yahoo\.com/quote/', url):
            content = self.yahoo_finance_agent.load_and_cache_data(url)
        else:
            return "The provided URL is not a supported Crunchbase or Yahoo Finance URL.", False

        if content:
            return f"Successfully loaded comprehensive data for this entity. You can now ask detailed questions!", True
        else:
            return f"Failed to load data from {url}.", False


    def answer_crunchbase_question(self, question: str, url: str, history: list):
        # ... same as before
        return self.crunchbase_agent.answer_question_from_context(question, url, history)

    def answer_yahoo_finance_question(self, question: str, url: str, history: list):
        """Answers a question about a Yahoo Finance entity."""
        return self.yahoo_finance_agent.answer_question_from_context(question, url, history)

    def find_and_recommend_product(self, user_query: str) -> str:
        # ... same as before
        print("\n1/3: Parsing user query...")
        params = self.llm.extract_search_parameters(user_query)
        if not params.get("keywords"):
            return "I couldn't figure out what product you're looking for."
        print("2/3: Searching for candidate products...")
        product_urls = search_amazon_for_product_urls(params, domain=self.amazon_domain_tld)
        if not product_urls:
            return f"I couldn't find any products on amazon.{self.amazon_domain_tld}."
        print(f"3/3: Gathering and analyzing details for {len(product_urls)} products...")
        all_product_data = []
        for p_url in product_urls:
            details = self.amazon_agent.get_product_details(p_url)
            if details:
                all_product_data.append(details)
        if not all_product_data:
            return "I found products but was unable to analyze their details."
        final_recommendation = self.llm.make_final_recommendation(user_query, all_product_data)
        return final_recommendation