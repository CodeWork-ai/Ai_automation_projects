import os

from dotenv import load_dotenv

from groq import Groq

# Load environment variables from .env file
load_dotenv()

def get_failure_reason_and_suggestion(status_code, response_text, api_type, endpoint_url):
    """
    Provides a rule-based reason and suggestion for common HTTP status codes.
    """
    # Handle None status_code
    if status_code is None:
        return ("Connection Error", "Unable to connect to the API endpoint. Check network connectivity and URL validity.")

    if status_code == 401:
        return ("Authentication Error", "The API key or token is invalid, expired, or missing. Check your credentials.")

    if status_code == 403:
        return ("Forbidden Error", "You do not have permission to access this resource. Check your API key's scope and permissions.")

    if status_code == 429:
        return ("Rate Limit Exceeded", "The application is making too many requests. Consider optimizing calls or requesting a higher limit.")

    if status_code >= 500:
        suggestion = (
            f"This is a server-side issue. For an '{api_type}' API, you should "
            f"{'check the server logs and application health.' if api_type == 'internal' else 'check the provider status page and contact their support.'}"
        )
        return ("Server-Side Error", suggestion)

    if status_code == 404:
        return ("Not Found", f"The endpoint URL '{endpoint_url}' does not exist. Verify the URL path and parameters.")

    return ("Unknown Error", f"Received status code {status_code}. Investigate the response body for more details.")

def get_llm_insights(error_context):
    """
    Uses Groq LLM to get advanced, human-readable insights into an API failure.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "Groq API key not found. Please set GROQ_API_KEY in your .env file."

    client = Groq(api_key=api_key)

    prompt = f"""
As an expert Site Reliability Engineer, analyze the following API failure and provide a concise, actionable suggestion.

**Error Context:**
- API Name: {error_context.get('name')}
- API Type: {error_context.get('api_type')} (This is important. 'internal' means we control the server, 'external' means it's a third-party service.)
- Endpoint URL: {error_context.get('endpoint')}
- HTTP Status Code: {error_context.get('status_code')}
- Response Body: {error_context.get('response_text')}

**Your Task:**
Based on the context, what is the most likely root cause, and what is the single most important next step the development team should take to triage this issue?

Be brief and direct.
"""

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.5,
            max_tokens=150
        )
        return chat_completion.choices[0].message.content

    except Exception as e:
        return f"Error contacting Groq API: {e}"
