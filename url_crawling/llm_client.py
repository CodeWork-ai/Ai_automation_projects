import json
import google.generativeai as genai
from config import GEMINI_API_KEY

class LLMClient:
    def __init__(self):
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is not set.")
        genai.configure(api_key=GEMINI_API_KEY)
        self.model = genai.GenerativeModel('gemini-1.5-flash')

    def ask_gemini(self, prompt: str):
        """Send query to Gemini and return text response."""
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Error with Gemini API: {e}")
            return "Unable to generate response from LLM."

    def extract_json(self, prompt: str):
        """Ask Gemini to extract information in JSON format."""
        try:
            response = self.model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
            return response.text
        except Exception as e:
            print(f"Error with Gemini API for JSON extraction: {e}")
            return "{}"

    def extract_search_parameters(self, user_query: str) -> dict:
        """Uses the LLM to extract structured search parameters from a user query."""
        prompt = f"""
        Analyze the user's request and extract search parameters as a JSON object.
        The JSON should have keys: "keywords", "brand", "min_price", and "max_price".
        - If a price is mentioned (e.g., "around 500"), create a reasonable range (e.g., 400 to 600).
        - If no value is found for a key, set it to null.

        User Request: "give me best wallet around 500 rupees."
        JSON Output: {{"keywords": "leather wallet", "brand": null, "min_price": 400, "max_price": 600}}

        User Request: "hp laptop under 60000"
        JSON Output: {{"keywords": "laptop", "brand": "HP", "min_price": null, "max_price": 60000}}
        ---
        User Request: "{user_query}"
        JSON Output:
        """
        try:
            response_text = self.extract_json(prompt)
            return json.loads(response_text)
        except (json.JSONDecodeError, Exception) as e:
            print(f"Error extracting search parameters: {e}")
            return {"keywords": user_query, "brand": None, "min_price": None, "max_price": None}

    def make_final_recommendation(self, user_query: str, product_data: list) -> str:
        """Takes product data and generates a final, structured recommendation."""
        all_product_str = json.dumps(product_data, indent=2)

        prompt = f"""
    You are an expert shopping advisor. Your task is to analyze the provided product data and recommend the best option based on the user's request.

    **User's Original Request:** "{user_query}"

    **Product Data (JSON):**
    {all_product_str}

    ---

    **Instructions for your response:**
    Generate a structured, easy-to-read recommendation using Markdown. Follow this format precisely:

    ### 🏆 Top Recommendation

    **Product:** [Product Name]
    - **Price:** [Current Price]
    - **Rating:** [Star Rating] (e.g., 4.5 out of 5 stars)
    - **Reason to Buy:** [A concise paragraph explaining why this is the best choice. Compare its features, price, and ratings against the other options and relate it back to the user's query.]

    ### ✨ Other Good Options

    **1. [Second Product Name]**
    - **Price:** [Current Price]
    - **Rating:** [Star Rating]
    - **Good for:** [Briefly explain why someone might choose this alternative. e.g., "A great budget-friendly choice."]

    **2. [Third Product Name]**
    - **Price:** [Current Price]
    - **Rating:** [Star Rating]
    - **Good for:** [Briefly explain why someone might choose this alternative. e.g., "Offers premium features for a higher price."]

    Do not add any introductory or concluding sentences outside of this structure.
    """
        return self.ask_gemini(prompt)

