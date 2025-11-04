import os
from groq import Groq
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize variables
client = None
model_is_available = False

# Get the API key from the environment
api_key = os.environ.get("GROQ_API_KEY")

if not api_key:
    print("CRITICAL ERROR: GROQ_API_KEY not found in .env file or environment.")
else:
    try:
        # Securely initialize the Groq client object
        client = Groq(api_key=api_key)
        model_is_available = True
        print("Groq client initialized successfully.")
    except Exception as e:
        print(f"Error initializing Groq client: {e}")


def generate_response(messages):
    """
    Generates a response from the Groq API.
    """
    if not model_is_available or not client:
        return "Groq client is not available. Please check your API key and terminal for errors."

    try:
        completion = client.chat.completions.create(
            # --- THE ONLY CHANGE IS HERE ---
            # Updated to a currently supported Llama 3.1 model
            model="llama-3.1-8b-instant", 
            messages=messages,
            temperature=0.7,
            max_tokens=2048,
            top_p=1,
            stream=False,
            stop=None
        )
        return completion.choices[0].message.content
    except Exception as e:
        error_message = f"An error occurred while communicating with the Groq API: {e}"
        print(error_message)
        return error_message