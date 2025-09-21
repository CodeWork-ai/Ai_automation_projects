# analyzer.py
import re
# --- CHANGE 1: Import 'Tuple' from the typing module ---
from typing import List, Dict, Any, Tuple
from groq_client import classify_requirement_with_groq, analyze_document_with_groq
import concurrent.futures

def run_full_analysis(text: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Orchestrates the entire analysis pipeline.

    --- CHANGE 2: The return type hint is now corrected to use Tuple[...] ---
    Returns:
        - A dictionary for the document-level overview.
        - A dictionary for the detailed sentence-level classification.
    """
    # 1. Document-Level Analysis
    document_overview = analyze_document_with_groq(text)
    
    if "error" in document_overview:
        return document_overview, {}

    # 2. Sentence-Level Analysis (only on identified requirements)
    functional_needs = document_overview.get("functional_needs", [])
    nfrs = document_overview.get("non_functional_requirements", [])
    
    all_requirements_sentences = functional_needs + nfrs
    
    detailed_specs = {
        "Functional": [],
        "Non-Functional": [],
        "Uncategorized": [],
        "Errors": []
    }

    if all_requirements_sentences:
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(classify_requirement_with_groq, all_requirements_sentences))
        
        for i, classification in enumerate(results):
            sentence = all_requirements_sentences[i]
            category = classification.get("category", "Functional") # Default to Functional
            
            # Refine category based on where it came from
            if sentence in nfrs:
                category = "Non-Functional"

            if category not in detailed_specs:
                detailed_specs[category] = []
            
            detailed_specs[category].append({
                "requirement": sentence,
                "subcategory": classification.get("subcategory", "N/A")
            })
            
    return document_overview, detailed_specs