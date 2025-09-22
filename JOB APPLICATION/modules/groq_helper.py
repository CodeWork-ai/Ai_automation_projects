import os
from groq import Groq
import streamlit as st
from groq import Groq

def get_groq_client():
    """Initializes and returns the Groq client."""
    api_key = os.getenv("GROQ_API_KEY")

    if api_key:
        # Strip quotes and whitespace if present
        api_key = api_key.strip().strip('"').strip("'")
    else:
        st.error("Groq API key is not configured. Please set it in your .env file.")
        return None

    return Groq(api_key=api_key)

def generate_response(prompt, model="llama-3.3-70b-versatile"):
    """Generates a response from the specified Groq model."""
    client = get_groq_client()
    if not client:
        return "Groq client could not be initialized."

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant for career services. Keep answers clear, professional, and well-structured."
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=model,
            temperature=0.7,
            max_tokens=2048,
            top_p=1,
            stop=None,
            stream=False,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        st.error(f"An error occurred with the Groq API: {e}")
        return None

def tailor_resume_prompt(resume_text, jd_text):
    """Creates a prompt for tailoring a resume with overall rating."""
    return f"""
    Based on the following resume and job description, provide suggestions on how to tailor the resume.
    - Each suggestion should have:
      1. A **bold heading**
      2. A short one-line description
    - Limit to 6–8 points.
    - At the end, provide an **overall evaluation of the resume** as one of: (Good), (Moderate), or (Needs Improvement).

    **My Resume:**
    ---
    {resume_text}
    ---

    **Job Description:**
    ---
    {jd_text}
    ---

    **Tailoring Suggestions (Bold Heading + Short Description):**
    - ...
    - ...

    **Overall Resume Evaluation: (Good / Moderate / Needs Improvement)**
    """

def draft_cover_letter_prompt(resume_text, jd_text):
    """Creates a prompt for drafting a shorter cover letter."""
    return f"""
    Based on the provided resume and job description, draft a **concise professional cover letter**.
    - Limit to 2 short paragraphs only (about half the length of a standard cover letter).
    - Highlight only the most relevant skills and experiences.
    - Keep it clear, direct, and engaging.

    **My Resume:**
    ---
    {resume_text}
    ---

    **Job Description:**
    ---
    {jd_text}
    ---

    **Concise Drafted Cover Letter (2 Short Paragraphs):**
    """

def generate_interview_questions_prompt(jd_text):
    """Creates a prompt for generating category-wise interview questions."""
    return f"""
    Based on the following job description, generate **only the most important interview questions**.
    - Organize them into multiple categories:
      1. Behavioral
      2. Technical
      3. Situational
      4. HR / General
      5. Problem-Solving
      6. Leadership
      7. Domain-Specific
    - Provide only 5–6 high-value questions per category.
    - Keep each question clear, concise, and relevant.

    **Job Description:**
    ---
    {jd_text}
    ---

    **Category-wise Important Interview Questions:**
    """
