# main_analyzer.py

import re
import json
from typing import List, Dict, Any
from groq_client import classify_requirement_with_groq
import concurrent.futures
from tqdm import tqdm # Import tqdm for the progress bar

def preprocess_input(text: str) -> List[str]:
    """Splits input text into sentences for classification."""
    if not isinstance(text, str):
        return []
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def initialize_specifications() -> Dict[str, List[Dict[str, str]]]:
    """Creates the initial structure for storing specifications."""
    return {
        "Functional": [],
        "Non-Functional": [],
        "Uncategorized": [],
        "Errors": []
    }

def generate_specifications_parallel(sentences: List[str], max_workers: int = 10) -> Dict[str, List[Dict[str, Any]]]:
    """
    Generates structured requirements by classifying sentences in parallel.
    
    Args:
        sentences: A list of sentences to classify.
        max_workers: The number of concurrent threads to use for API calls.
    """
    specs = initialize_specifications()
    print(f"Found {len(sentences)} sentences to classify. Starting parallel processing...")

    # Using ThreadPoolExecutor to make API calls concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # The 'map' function sends each sentence to the classifier function.
        # tqdm is wrapped around it to create a live progress bar.
        results = list(tqdm(executor.map(classify_requirement_with_groq, sentences), total=len(sentences), desc="Classifying Sentences"))
    
    # Now that all API calls are complete, we process the results
    for i, classification in enumerate(results):
        sentence = sentences[i]
        category = classification.get("category", "Uncategorized")
        
        if category not in specs:
            specs[category] = []

        specs[category].append({
            "requirement": sentence,
            "subcategory": classification.get("subcategory", "N/A")
        })
            
    return specs

def main():
    """Main function to run the requirement analyzer."""
    client_input = """
    The system should allow users to log in with their email and password.
    We need the application to be really fast and responsive, especially on mobile.
    The user must be able to upload a profile picture and edit their personal details.
    Security is a top priority; we need to ensure that all user data is encrypted.
    Also, the website should be available 24/7 without any downtime.
    The admin needs a dashboard to view monthly sales reports.
    The application must support up to 10,000 concurrent users without performance degradation.
    The user interface should be intuitive for non-technical users.
    All password data must be hashed, not stored in plain text.
    The system must be able to integrate with a third-party payment gateway like Stripe.
    """

    print("--- Starting Intelligent Requirement Analyzer (High-Efficiency Mode) ---")
    
    sentences_to_classify = preprocess_input(client_input)
    
    if not sentences_to_classify:
        print("No sentences found to analyze.")
        return
        
    structured_specs = generate_specifications_parallel(sentences_to_classify)
    
    print("\n--- Analysis Complete ---")
    print(json.dumps(structured_specs, indent=4))

if __name__ == "__main__":
    main()