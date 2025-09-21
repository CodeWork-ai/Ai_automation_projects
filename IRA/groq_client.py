# groq_client.py

import os
import json
from groq import Groq
from dotenv import load_dotenv
from typing import Dict, Any

# Load environment variables from .env file
load_dotenv()

# Get the API key from the environment
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY not found. Make sure it is set in your .env file.")

client = Groq(api_key=api_key)

def classify_requirement_with_groq(sentence: str) -> Dict[str, str]:
    """
    Classifies a single requirement sentence into Functional/Non-Functional categories.
    This function is used for the "Detailed Requirements Breakdown" tab.
    """
    system_prompt = """
    You are an expert requirement analyst. Your task is to classify a given software requirement sentence into one of two main categories: 'Functional' or 'Non-Functional'.
    
    Also, classify it into a specific subcategory from the following lists:
    - Functional Subcategories: 'User Authentication', 'Data Processing', 'User Interface', 'Reporting', 'External Interfaces'.
    - Non-Functional Subcategories: 'Performance', 'Security', 'Usability', 'Availability', 'Scalability', 'Maintainability'.
    
    If the sentence doesn't fit any category, classify it as 'Uncategorized' with a subcategory of 'General'.
    
    Your response MUST be a JSON object with two keys: "category" and "subcategory".
    For example: {"category": "Functional", "subcategory": "User Authentication"}
    """
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Classify the following requirement: '{sentence}'"},
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        return json.loads(chat_completion.choices[0].message.content)
    except Exception as e:
        print(f"Sentence classification failed: {e}")
        return {"category": "Errors", "subcategory": "API Error"}

def analyze_document_with_groq(full_text: str) -> Dict[str, Any]:
    """
    Analyzes the entire document from a Solutions Architect's perspective to produce the structured,
    technical output needed for the main project dashboard.
    """
    system_prompt = """
    You are a Senior Solutions Architect and a Principal Product Manager. Your task is to transform a high-level project brief into a structured, technically-grounded preliminary specification document. Your analysis must be detailed and specific.

    For each of the 10 categories below, provide a detailed analysis. The output MUST be a single JSON object.

    **Instructions for each category:**
    1.  `business_context_and_goals`: (List of objects with 'point' and 'sources') Summarize the core business problem and the primary goals.
    2.  `stakeholders_and_users`: **(List of objects with 'role', 'needs', and 'sources')** Identify user roles (e.g., 'Admin', 'Employee'), describe their primary needs in a short phrase, and cite the sources.
    3.  `functional_needs`: (List of objects with 'point' and 'sources') Structure each 'point' as a user story (e.g., "As an [user role], I want to [action], so that [benefit]").
    4.  `non_functional_requirements`: (List of objects with 'point' and 'sources') Quantify them where possible (e.g., 'API response time must be under 200ms', 'Uptime must be > 99.9%').
    5.  `constraints`: (List of objects with 'point' and 'sources') Identify any limitations like time, budget, or compliance.
    6.  `data_requirements`: (List of objects with 'point' and 'sources') Describe the key data entities and their likely relationships. (e.g., "An 'Employee' has many 'Tasks'.").
    7.  `technical_mapping`: **(OBJECT)** Infer a potential technical architecture with keys `"inferred_stack"` (an object) and `"components"` (a list).
    8.  `success_metrics`: (List of objects with 'point' and 'sources') Distinguish between business metrics and technical metrics.
    9.  `risks_and_assumptions`: (List of objects with 'point' and 'sources') What are the main technical or project risks?
    10. `prioritization_and_dependencies`: (List of objects with 'point' and 'sources') What is the likely MVP?

    A single source sentence can be used for multiple categories. Be thorough. If information is not present for a key, return an empty list `[]` for list-based keys or an empty object `{}` for `technical_mapping`.
    """
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Analyze the following document:\n\n{full_text}"},
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.3, # Slightly increased to encourage more detailed inference
            response_format={"type": "json_object"},
        )
        return json.loads(chat_completion.choices[0].message.content)
    except Exception as e:
        print(f"Document-level analysis failed: {e}")
        return {"error": "Failed to analyze the document. The content might be too complex or the API is unavailable."}