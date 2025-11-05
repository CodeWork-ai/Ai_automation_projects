# llm_manager.py

import os
import re
import random
from dotenv import load_dotenv
from pathlib import Path
from groq import Groq


load_dotenv()

class WeatherLLM:
    def __init__(self, model_name="llama-3.3-70b-versatile"):
        try:
            # Load environment variables
            env_path = Path('.') / '.env'
            if env_path.exists():
                load_dotenv(dotenv_path=env_path)
            
            self.model_name = model_name
            self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
            print(f"Initialized Groq client for model: {model_name}")

        except Exception as e:
            print(f"Error initializing WeatherLLM with Groq: {str(e)}")
            self.client = None

    def __call__(self, prompt, max_tokens=1024, temperature=0.7):
        if self.client is None:
            # Fallback response
            return "I'm having trouble generating a response right now."
            
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model=self.model_name,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            print(f"Error generating response from Groq: {str(e)}")
            return "I'm having trouble generating a response right now."

    def analyze_sentiment(self, text):
        # This can be improved with a dedicated sentiment analysis model if needed
        positive_words = ['good', 'great', 'excellent', 'nice', 'beautiful', 'sunny', 'clear']
        negative_words = ['bad', 'terrible', 'awful', 'rainy', 'stormy', 'cold', 'hot']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            return "POSITIVE", 0.7
        elif negative_count > positive_count:
            return "NEGATIVE", 0.3
        else:
            return "NEUTRAL", 0.5

    def draft_follow_up(self, context, sentiment):
        # The main prompt will now handle suggestions, making this less critical
        # but can be kept for fallback.
        follow_ups = [
            "Is there anything else I can help you with?",
            "Would you like a forecast for another location?",
            "Can I provide more details on the weather?",
        ]
        return random.choice(follow_ups)

    def generate_conversational_response(self, context: str, sentiment: str) -> str:
        """Generate a conversational response using the Groq LLM"""
        if self.client is None:
            return self.generate_fallback_response("", context)
            
        try:
            # The context is now a very detailed prompt from the 'reason' node
            response = self(context)
            return response
        except Exception as e:
            print(f"Error generating conversational response: {str(e)}")
            return self.generate_fallback_response("", context)
    
    def generate_fallback_response(self, user_input, weather_info):
        """Generate a simple rule-based response when the LLM fails"""
        # This remains as a safety net
        user_input = user_input.lower()
        
        temp_match = re.search(r'Temperature:\s*([\d.]+)\s*°C', weather_info)
        condition_match = re.search(r'Condition:\s*([^\n]+)', weather_info)
        humidity_match = re.search(r'Humidity:\s*([\d.]+)%', weather_info)
        
        temp = temp_match.group(1) if temp_match else "unknown"
        condition = condition_match.group(1).strip() if condition_match else "unknown"
        humidity = humidity_match.group(1) if humidity_match else "unknown"
        
        if "rain" in user_input:
            if "rain" in condition.lower():
                return f"Yes, it appears to be raining. The temperature is {temp}°C."
            else:
                return f"No, it doesn't seem to be raining. The current condition is {condition}."
        
        return f"The current weather is {condition} with a temperature of {temp}°C and {humidity}% humidity."