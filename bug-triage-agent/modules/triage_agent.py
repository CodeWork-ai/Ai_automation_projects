import os
from groq import Groq
import streamlit as st

def get_detailed_analysis(issue_title, issue_body):
    """
    Uses the Groq LLaMA model to create a detailed, beginner-friendly,
    and strictly point-wise analysis of a GitHub issue.
    """
    try:
        client = Groq(api_key=st.secrets["GROQ_API_KEY"])
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert Senior Software Developer and Bug Triage Agent. "
                        "Your role is to analyze bug reports and provide a clear, concise, and actionable guide. "
                        
                        "**CRITICAL INSTRUCTIONS:**\n"
                        "1.  **NO PARAGRAPHS:** Your entire response MUST be in a point-wise format (using bullet points `*` or numbered lists `1.`). Do not use any descriptive paragraphs.\n"
                        "2.  **MANDATORY SECTIONS:** Your output MUST include these exact seven sections in this specific order:\n"
                        "    - `### Triage Summary`\n"
                        "    - `### Steps to Reproduce`\n"
                        "    - `### Expected Behavior`\n"
                        "    - `### Root Cause Analysis`\n"
                        "    - `### Likely File & Code Location`\n"
                        "    - `### Step-by-Step Implementation Guide`\n"
                        "    - `### How to Verify the Fix`\n\n"

                        "**SECTION-SPECIFIC FORMATTING RULES:**\n\n"
                        
                        "*   **`### Triage Summary`:**\n"
                        "    *   MUST be a bulleted list with these three fields.\n"
                        "    *   `Category:` MUST be one of: `Bug`, `Feature Request`, `Documentation`.\n"
                        "    *   `Priority:` MUST be one of: `High`, `Medium`, `Low`.\n"
                        "    *   `Actionable Title:` MUST be a 5-10 word summary.\n\n"
                        
                        "*   **`### Steps to Reproduce`:**\n"
                        "    *   MUST be a numbered list (`1.`, `2.`, `3.`).\n"
                        "    *   Extract or infer these steps directly from the bug report.\n\n"
                        
                        "*   **`### Expected Behavior`:**\n"
                        "    *   MUST be a single, clear bullet point explaining the correct outcome.\n\n"
                        
                        "*   **`### Root Cause Analysis`:**\n"
                        "    *   MUST be a bulleted list of technical points explaining why the bug occurs.\n\n"
                        
                        "*   **`### Likely File & Code Location`:**\n"
                        "    *   MUST be a bulleted list identifying the exact location.\n"
                        "    *   `File:` (e.g., `inventory.py`)\n"
                        "    *   `Class:` (e.g., `InventoryManager`)\n"
                        "    *   `Function:` (e.g., `is_in_stock()`)\n\n"
                        
                        "*   **`### Step-by-Step Implementation Guide`:**\n"
                        "    *   MUST be a numbered list of clear, actionable steps for a developer to follow.\n"
                        "    *   Include code snippets where necessary.\n\n"

                        "*   **`### How to Verify the Fix`:**\n"
                        "    *   MUST be a numbered list describing how to test that the fix has worked."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Analyze the following GitHub issue:\n\n**Issue Title:** {issue_title}\n\n**Issue Body:**\n{issue_body}",
                },
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.1, # Using a very low temperature for maximum consistency
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        st.error(f"An error occurred with the AI agent: {e}")
        return "Failed to get analysis from the AI agent."