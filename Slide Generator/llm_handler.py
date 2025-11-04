import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

def generate_presentation_structure(prompt_text):
    """
    Definitive version: Uses Groq's JSON mode with a highly-structured and strict prompt
    that enforces a 5-point limit per slide and a minimum of 5 content slides.
    """
    print("\n\n>>> RUNNING THE LATEST llm_handler.py with 5-CONTENT-SLIDE MINIMUM <<<\n\n")

    try:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables.")
            
        client = Groq(api_key=api_key)

        # Updated prompt with the critical 5-content-slide minimum rule.
        system_prompt = """
        You are an expert content strategist. Your task is to generate a presentation outline.
        You MUST output a single, valid JSON object. Do NOT output any other text or markdown.

        This object must have a single top-level key: "slides".
        The value of "slides" MUST be an array of slide objects.

        Each object in the "slides" array must have two keys: "slide_type" and "content".
        The "content" key's value MUST be an object.

        Here are the strict rules for each "slide_type":
        1. If "slide_type" is "title_slide", the "content" object MUST have "title" and "subtitle" keys.
        2. If "slide_type" is "thank_you_slide", the "content" object MUST have a "title" key.
        3. If "slide_type" is "content_slide", the "content" object MUST have a "title" key and a "bullet_points" key.
           - The "bullet_points" key MUST be an array of strings.
           - This array MUST NOT contain more than 5 items.
           - If a topic has more than 5 points, you MUST split it into multiple 'content_slide' objects, each with a maximum of 5 points. For example, use titles like "Topic (Part 1)" and "Topic (Part 2)".
        
        CRITICAL RULE: The final JSON array for "slides" MUST contain a minimum of 5 objects where the "slide_type" is "content_slide". You must generate enough detailed content to meet this requirement, covering different aspects of the user's topic.
        """

        chat_completion = client.chat.completions.create(
            messages=[
                { "role": "system", "content": system_prompt },
                { "role": "user", "content": f"Create a presentation structure based on this input. Start with a 'title_slide' and end with a 'thank_you_slide'. Strictly follow all rules, including the 5-bullet-point-per-slide rule and the minimum of 5 content slides. Input: '{prompt_text}'" }
            ],
            model="openai/gpt-oss-120b",
            temperature=0.7,
            max_tokens=4096,
            response_format={"type": "json_object"}, 
        )

        response_text = chat_completion.choices[0].message.content
        response_data = json.loads(response_text)
        slide_list = response_data.get("slides")

        if not slide_list or not isinstance(slide_list, list):
            print("Error: The AI did not return a valid 'slides' array.")
            return None
            
        return slide_list

    except Exception as e:
        print(f"An error occurred in llm_handler: {e}")
        return None