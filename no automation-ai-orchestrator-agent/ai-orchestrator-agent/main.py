# main.py
import json
import re
from tools import knowledge_base, jira_api, github_api

from llm_client import get_ai_analysis_and_plan

def parse_llm_response(response: str):
    """Parses the structured Markdown response from the LLM into a list of solutions."""
    solutions = []
    try:
        # Split the response into a root cause part and the solutions part
        parts = re.split(r'### Solutions', response, flags=re.IGNORECASE)
        root_cause_analysis = parts[0].replace("### Root Cause Analysis", "").strip()
        
        # Split the solutions part by the solution header
        solution_blocks = re.split(r'#### Solution \d+:', parts[1])
        
        for block in solution_blocks:
            if not block.strip():
                continue

            solution = {}
            # Use regex to find the JSON action block first
            action_match = re.search(r"```json\s*(\{.*?\})\s*```", block, re.DOTALL)
            if action_match:
                solution['action'] = json.loads(action_match.group(1))
                # Remove the action block for easier parsing of other fields
                block = block.replace(action_match.group(0), "")

            # Parse other fields
            desc_match = re.search(r"Description:(.*?)(Pros:|Cons:|Confidence Score:)", block, re.DOTALL)
            pros_match = re.search(r"Pros:(.*?)(Cons:|Confidence Score:)", block, re.DOTALL)
            cons_match = re.search(r"Cons:(.*?)(Confidence Score:)", block, re.DOTALL)
            score_match = re.search(r"Confidence Score:\s*(\d+)%", block, re.DOTALL)

            if desc_match: solution['description'] = desc_match.group(1).strip()
            if pros_match: solution['pros'] = pros_match.group(1).strip()
            if cons_match: solution['cons'] = cons_match.group(1).strip()
            if score_match: solution['confidence'] = int(score_match.group(1))
            
            solutions.append(solution)
            
        return root_cause_analysis, solutions
    except (IndexError, json.JSONDecodeError, AttributeError) as e:
        print(f"\n--- ERROR PARSING LLM RESPONSE ---")
        print(f"Error: {e}")
        print("LLM Output may not be in the expected format. Raw response:")
        print(response)
        print("---------------------------------\n")
        return "Could not parse analysis.", []


def main_workflow():
    """The main end-to-end workflow for the AI Knowledge Orchestrator."""
    
    # 1. TRIGGER
    print("--- STEP 1: TRIGGER ---")
    trigger = "High latency detected on the user authentication service after the latest deployment. Users are reporting 500 errors during login."
    parent_jira_ticket = "PROD-123"
    print(f"🔥 New Trigger Received (from {parent_jira_ticket}): {trigger}\n")

    # 2. CONTEXT GATHERING
    print("--- STEP 2: CONTEXT GATHERING ---")
    code_context = knowledge_base.get_code_context(trigger)
    docs_context = knowledge_base.get_docs_context(trigger)
    full_context = f"Code Context:\n{code_context}\n\nDocumentation Context:\n{docs_context}"
    print("✅ Context gathered.\n")

    # 3. AI RESEARCH & SYNTHESIS
    print("--- STEP 3: AI RESEARCH & SYNTHESIS ---")
    llm_response = get_ai_analysis_and_plan(trigger, full_context)
    root_cause, solutions = parse_llm_response(llm_response)

    if not solutions:
        print("Could not proceed. Analysis failed.")
        return

    print("\n✅ DevBrain Analysis Complete:")
    print("===================================")
    print("ROOT CAUSE ANALYSIS:")
    print(root_cause)
    print("===================================\n")


    # 4. FACILITATE CONSENSUS
    print("--- STEP 4: FACILITATE CONSENSUS ---")
    print("Proposed Solutions:")
    for i, sol in enumerate(solutions):
        print(f"[{i+1}] {sol.get('description', 'N/A')} (Confidence: {sol.get('confidence', 0)}%)")
        print(f"    Pros: {sol.get('pros', 'N/A')}")
        print(f"    Cons: {sol.get('cons', 'N/A')}")
        print(f"    Action: {sol.get('action', 'N/A')}\n")

    try:
        choice = int(input("Please select the best solution to execute (1, 2, 3): ")) - 1
        if not 0 <= choice < len(solutions):
            raise ValueError
        chosen_solution = solutions[choice]
    except (ValueError, IndexError):
        print("Invalid choice. Aborting.")
        return

    print(f"\n✅ Team consensus reached. Executing option {choice + 1}.\n")

    # 5. ORCHESTRATE ACTION
    print("--- STEP 5: ORCHESTRATE ACTION ---")
    action = chosen_solution.get('action', {})
    tool = action.get('tool')
    params = action.get('params', {})

    if tool == 'github':
        github_api.trigger_workflow(params.get('workflow_name'), params.get('parameters'))
    elif tool == 'jira':
        jira_api.create_sub_task(
            parent_ticket=parent_jira_ticket,
            title=params.get('title'),
            body=params.get('body')
        )
    else:
        print("No valid action found to execute.")

    # 6. (SIMULATED) LEARNING
    print("\n--- STEP 6: LEARNING ---")
    print("Outcome recorded to knowledge base for future reference.")


if __name__ == "__main__":
    main_workflow()