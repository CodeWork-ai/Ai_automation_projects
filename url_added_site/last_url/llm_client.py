import json
from config import GROQ_API_KEY
from groq import Groq

class LLMClient:
    # Note: I've changed the model back to a valid Groq model.
    # "openai/gpt-oss-120b" is not available on the Groq platform.
    def __init__(self, model="llama-3.3-70b-versatile"):
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is not set.")
        self.client = Groq(api_key=GROQ_API_KEY)
        self.model = model

    def _get_completion(self, prompt: str, is_json: bool = False):
        """Send a prompt to the Groq API and return the text response."""
        messages = [{"role": "user", "content": prompt}]
        try:
            # Add response_format parameter if JSON is expected
            response_format = {"type": "json_object"} if is_json else None
            
            chat_completion = self.client.chat.completions.create(
                messages=messages,
                model=self.model,
                temperature=0.7,
                max_tokens=8000,
                response_format=response_format
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            print(f"Error with Groq API: {e}")
            return "{}" if is_json else "Unable to generate response from LLM."

    def ask_llm(self, prompt: str) -> str:
        """
        A general-purpose method to get a text response from the LLM.
        This is the replacement for the old 'ask_gemini' method.
        """
        return self._get_completion(prompt, is_json=False)

    def extract_search_parameters(self, user_query: str) -> dict:
        """Uses the LLM to extract structured search parameters from a user query."""
        prompt = f"""
        Analyze the user's request and extract search parameters as a JSON object.
        The JSON should have keys: "keywords", "brand", "min_price", and "max_price".
        - You must respond in a valid JSON format.
        ---
        User Request: "{user_query}"
        JSON Output:
        """
        try:
            response_text = self._get_completion(prompt, is_json=True)
            return json.loads(response_text)
        except (json.JSONDecodeError, Exception) as e:
            print(f"Error extracting search parameters: {e}")
            return {"keywords": user_query, "brand": None, "min_price": None, "max_price": None}
    
    def extract_product_details_json(self, prompt: str) -> str:
        """Uses the LLM to extract product details in JSON format."""
        return self._get_completion(prompt, is_json=True)
    

# In llm_client.py, inside the LLMClient class

    def make_final_recommendation(self, user_query: str, product_data: list) -> str:
        """Takes product data and generates a final, structured recommendation."""
        all_product_str = json.dumps(product_data, indent=2)

        # --- MODIFY THE PROMPT ---
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
        - **Link:** [Product URL]  # <-- ADDED THIS LINE
        - **Reason to Buy:** [A concise paragraph explaining why this is the best choice...]

        ### ✨ Other Good Options

        **1. [Second Product Name]**
        - **Price:** [Current Price]
        - **Rating:** [Star Rating]
        - **Link:** [Product URL]  # <-- ADDED THIS LINE
        - **Good for:** [Briefly explain why someone might choose this alternative...]

        **2. [Third Product Name]**
        - **Price:** [Current Price]
        - **Rating:** [Star Rating]
        - **Link:** [Product URL]  # <-- ADDED THIS LINE
        - **Good for:** [Briefly explain why someone might choose this alternative...]

        Do not add any introductory or concluding sentences outside of this structure.
        """
        # --- END OF MODIFICATION ---
        return self._get_completion(prompt)


