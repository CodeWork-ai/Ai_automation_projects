import os
import json
from groq import Groq
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize the Groq client
client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

def extract_topics_from_text(context):
    """
    Analyzes text to identify and extract main topics with their corresponding content.
    """
    if not context:
        return "{}"

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": """
                You are a topic extraction expert. Your task is to analyze the provided text and break it down into its main topics or sections.
                You MUST respond with ONLY a valid JSON object.
                This JSON object should have keys that are the topic titles (e.g., "Introduction to AI", "Search Algorithms").
                The value for each key must be the full text content associated with that topic.
                Do not include topics with less than 50 words of content.
                Example:
                {
                  "Topic 1 Title": "All the text content related to topic 1...",
                  "Topic 2 Title": "All the text content related to topic 2..."
                }
                """
            },
            {
                "role": "user",
                "content": f"Here is the study material. Please extract the topics and their content from it:\n---\n{context}\n---",
            }
        ],
        model="llama-3.3-70b-versatile",
        response_format={"type": "json_object"},
    )
    return chat_completion.choices[0].message.content

def simplify_and_structure_text(context):
    """
    Analyzes the full text and provides a structured, simplified summary.
    """
    if not context:
        return "The document appears to be empty. Please upload a document with text."

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": """
                You are Elley AI, an expert teacher who makes complex topics easy and fun to understand.
                Your task is to analyze the provided study material and break it down into a structured, engaging summary.
                You MUST use the following structure and formatting:
                
                ### 🔍 Basic Introduction
                - Start with a simple, clear overview of the main topic.
                
                ### 🚀 Use Cases
                - List the primary applications or purposes of the concept in bullet points.
                
                ### 🌍 Real-world Applications
                - Provide specific, real-world examples that the user can relate to.
                
                ### 👍 Advantages & 👎 Disadvantages
                - Create two sub-sections.
                - Use bullet points to list the key pros and cons.
                
                ### 🔗 Connections to Other Concepts
                - Explain how this topic connects or is used in other fields or concepts.
                
                ### 💡 Memorable Clue
                - Provide a funny or memorable analogy, acronym, or short story to help the user remember the core idea.
                
                Your entire response must follow this structure. Use emojis, bullet points, and bold text to make it easy to read. DO NOT write long, continuous paragraphs.
                """
            },
            {
                "role": "user",
                "content": f"Here is my study material. Please analyze it fully and simplify it for me:\n---\n{context}\n---",
            }
        ],
        model="llama-3.3-70b-versatile",
    )
    return chat_completion.choices[0].message.content

def generate_summary(context, summary_type):
    """
    Generates a summary of the provided text based on the specified type.
    """
    if not context:
        return "The document is empty. Please upload a document with text to summarize."

    if summary_type == "Quick Overview":
        instruction = "Create a very concise, high-level summary in one short paragraph. Focus only on the main topic and conclusion."
    elif summary_type == "Detailed Summary":
        instruction = "Create a detailed summary. It should be several paragraphs long, covering the key arguments, evidence, and findings in the document."
    elif summary_type == "Key Points Breakdown":
        instruction = "Analyze the document to identify the main sections or chapters. For each section, provide a heading and a bulleted list of its most important points. If the document is not explicitly structured, create logical sections yourself."
    else:
        return "Invalid summary type selected."

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are an expert academic summarizer. Your task is to generate a summary of the user's study material based on their desired format. You must follow the user's instructions precisely."
            },
            {
                "role": "user",
                "content": f"Here is my study material:\n---\n{context}\n---\nMy request is: {instruction}",
            }
        ],
        model="llama-3.3-70b-versatile",
    )
    return chat_completion.choices[0].message.content

def chat_with_elley(prompt, context):
    """
    Acts as a chatbot, simplifying or answering questions based on context.
    """
    if not prompt:
        return "Please ask a question."

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are Elley AI, a friendly and expert study partner. Your goal is to help users understand their study material. "
                           "If the user asks to simplify or explain something, rewrite it clearly with analogies and examples. "
                           "If the user asks a specific question, answer it based ONLY on the provided context material. "
                           "If the answer isn't in the material, say so politely. Be encouraging and supportive."
            },
            {
                "role": "user",
                "content": f"Here is my study material:\n---\n{context}\n---\nMy request is: {prompt}",
            }
        ],
        model="llama-3.3-70b-versatile",
    )
    return chat_completion.choices[0].message.content

def answer_question_from_text(context, question):
    """
    Uses the Groq API to answer a question based on a specific context text (for Note Lens).
    """
    if not context or not question:
        return "Please provide both the text from your material and your question."

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful study assistant. Based ONLY on the provided text, answer the user's question. If the answer is not in the text, say 'The answer is not found in the provided text.'"
            },
            {
                "role": "user",
                "content": f"Here is the material from a specific page:\n---\n{context}\n---\nMy question is: {question}",
            }
        ],
        model="llama-3.3-70b-versatile",
    )
    return chat_completion.choices[0].message.content

def generate_quiz_from_text(topic_text, num_questions, quiz_type):
    """
    Uses the Groq API to generate a quiz in a structured JSON format.
    """
    if not topic_text:
        return "Please provide some text to generate a quiz from."

    if quiz_type == "Multiple Choice":
        format_instructions = """
        Each object must contain:
        - "type": "multiple_choice"
        - "question": The question text (string).
        - "options": A list of exactly 4 unique strings.
        - "answer": The full string of the correct option, which must be an exact match to one of the strings in the "options" list.
        """
    else: # Flashcards
        format_instructions = """
        Each object must contain:
        - "type": "flashcard"
        - "question": The question or term to define (string).
        - "answer": The correct answer or definition (string).
        """

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": f"You are a quiz generator. Create a {num_questions}-question quiz based on the provided text. "
                           f"The quiz type is '{quiz_type}'. "
                           "You MUST respond with ONLY a valid JSON object containing a single key 'questions' which holds a list of question objects. "
                           f"Each object in the list represents one question. Adhere strictly to this format: {format_instructions}"
            },
            {
                "role": "user",
                "content": f"Generate the quiz from this text:\n---\n{topic_text}\n---",
            }
        ],
        model="llama-3.3-70b-versatile",
        response_format={"type": "json_object"},
    )
    return chat_completion.choices[0].message.content

def get_brief_explanation(question, correct_answer, context):
    """
    Generates a brief, one-paragraph explanation for a quiz answer.
    """
    prompt = f"""
    Based on the provided context, briefly explain why the correct answer to the question is what it is.
    Keep the explanation to a single, concise paragraph.

    **Context Material:**
    ---
    {context}
    ---
    
    **Question:** {question}
    **Correct Answer:** {correct_answer}
    
    Your explanation should be clear and directly address the question and the correct answer.
    """
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are an expert tutor AI. Your goal is to provide a brief and clear explanation for a quiz answer based on the provided context."
            },
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="llama-3.3-70b-versatile",
    )
    return chat_completion.choices[0].message.content

def get_concept_explanation(question, user_answer, correct_answer, context):
    """
    Generates multi-level explanations for incorrect MCQ answers.
    """
    prompt = f"""
    A student answered a multiple-choice question incorrectly. Your task is to provide a multi-level explanation to help them understand the concept better.
    
    **Original Context Material:**
    ---
    {context}
    ---
    
    **Question:** {question}
    **Student's Incorrect Answer:** {user_answer}
    **Correct Answer:** {correct_answer}
    
    Please generate an explanation with three distinct levels of depth, following this structure exactly:
    
    #### 🤔 Why the Answer is Correct
    
    **Level 1: Beginner**
    - Provide a very simple, one-sentence explanation using an everyday analogy.
    
    **Level 2: Intermediate**
    - Explain the concept in more detail, referring directly to the core idea from the context material.
    
    **Level 3: Advanced**
    - Give a more technical or nuanced explanation, connecting it to broader concepts or implications.
    """
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are an expert tutor AI. Your goal is to explain concepts clearly and at multiple levels to help students who have answered a question incorrectly. You must follow the user-provided structure precisely."
            },
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="llama-3.3-70b-versatile",
    )
    return chat_completion.choices[0].message.content


def analyze_weak_areas(wrong_questions_details, context):
    """
    Analyzes patterns in wrong answers and generates targeted, concise feedback.
    """
    wrong_questions_str = "\n".join([f"- Question: {q['question']} (Correct Answer: {q['answer']})" for q in wrong_questions_details])

    prompt = f"""
    A student has answered the following questions incorrectly. Your task is to identify their weak areas and provide short, targeted suggestions.

    **Incorrectly Answered Questions:**
    {wrong_questions_str}

    Please provide your analysis using the following structure. BE VERY CONCISE. Do not use long paragraphs.

    ### 📉 Your Weak Areas & Key Suggestions
    - Based on the wrong answers, identify 1-3 specific topics the student is struggling with.
    - For each topic, provide just 1-2 bullet points with a very brief, core-concept reminder.
    - Example: `- **CrunchbaseAgent:** Remember, its main job is fetching and caching data from Crunchbase.`

    ### 🎯 Actionable Next Steps
    - Suggest 2-3 specific and short actions the student can take.
    - Example: `- Review the 'Agents' section in your notes.` or `- Practice differentiating between LLMClient and BaseAgent.`
    """
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are an AI academic advisor. Your job is to analyze a student's incorrect quiz answers, identify conceptual weak spots, and provide a constructive, actionable, and VERY CONCISE improvement plan. Follow the user's requested structure precisely."
            },
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="llama-3.3-70b-versatile",
    )
    return chat_completion.choices[0].message.content

def find_uncovered_topics(all_topic_titles, quiz_questions, context):
    """
    Analyzes a quiz and a list of topics to find which topics were not covered.
    """
    quiz_questions_str = json.dumps(quiz_questions, indent=2)
    all_topics_str = ", ".join(all_topic_titles)

    prompt = f"""
    You are an expert content analyst. Your task is to determine which topics from a provided list are not covered by a given quiz.
    
    **Full List of Document Topics:**
    [{all_topics_str}]
    
    **Generated Quiz Questions:**
    {quiz_questions_str}
    
    **Full Document Context:**
    ---
    {context}
    ---
    
    Analyze the quiz questions and the full document context. Based on your analysis, identify which topics from the list are NOT represented in the quiz questions.
    
    You MUST respond with ONLY a valid JSON object. This object should contain a single key, "uncovered_topics", which is a list of strings. Each string in the list should be the title of an uncovered topic.
    If all topics are covered, return an empty list.
    
    Example response:
    {{
      "uncovered_topics": ["Topic Title A", "Topic Title C"]
    }}
    """
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are an expert content analyst. Your job is to find topics that are not covered in a quiz. You must respond in the specified JSON format."
            },
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="llama-3.3-70b-versatile",
        response_format={"type": "json_object"},
    )
    try:
        response_json = json.loads(chat_completion.choices[0].message.content)
        return response_json.get("uncovered_topics", [])
    except (json.JSONDecodeError, AttributeError):
        return []