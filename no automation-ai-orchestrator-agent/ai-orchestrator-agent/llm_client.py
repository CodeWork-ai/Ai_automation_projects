# llm_client.py
import json
from groq import Groq
import config

# Initialize the Groq client
try:
    client = Groq(api_key=config.GROQ_API_KEY)
except Exception as e:
    print(f"Error initializing Groq client: {e}")
    client = None

def get_ai_analysis_and_plan(trigger: str, context: str) -> str:
    """
    Uses Groq's Llama3 model to analyze an issue and propose actionable solutions.
    """
    if not client:
        return "Groq client not initialized."

    # UPDATED: This prompt now strictly requires a minimum of 3 solutions AND a valid Action block for each.
    system_prompt = """
    You are "DevBrain," an expert AI Knowledge Orchestrator. Your response MUST follow the specified format exactly. Do not add any conversational text or preambles.

    **Your Task:**
    Analyze the provided Trigger and Context to determine the root cause of an IT issue and propose a minimum of 3 distinct, actionable solutions.

    **RESPONSE FORMAT (Strict):**
    You MUST structure your response using the following Markdown template. Do not deviate. Every solution you propose MUST include a valid `Action` JSON block, even if the action is to create a simple investigation issue.

    ### Root Cause Analysis
    [Your detailed analysis of the likely root cause goes here. Be concise.]

    ### Solutions
    #### Solution 1: [Title of Solution 1]
    **Description:** [Detailed description of the solution.]
    **Pros:** [Benefits of this approach.]
    **Cons:** [Drawbacks or risks.]
    **Confidence Score:** [A number from 0 to 100, followed by a single percent sign (e.g., "95%").]
    **Action:**
    ```json
    {
      "tool": "github",
      "function": "[function_name]",
      "params": { ... }
    }
    ```

    #### Solution 2: [Title of Solution 2]
    [...Follow the same structure...]

    #### Solution 3: [Title of Solution 3]
    [...Follow the same structure...]

    **Available GitHub Functions:**
    - `trigger_workflow`: For automated actions. `params` should include `workflow_name` and `parameters`.
    - `create_github_issue`: For manual investigation. `params` should include `repo`, `title`, and `body`.

    Begin your response immediately with "### Root Cause Analysis".
    """

    user_prompt = f"""
    **Trigger:**
    {trigger}

    **Collected Context:**
    {context}
    """

    print("\n🧠 Contacting DevBrain (Groq Llama3)... Please wait.")

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model=config.MODEL_NAME,
            temperature=0.5, 
            max_tokens=3072, 
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"An error occurred with the Groq API: {e}"