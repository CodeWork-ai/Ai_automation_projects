import streamlit as st
import json
import re
from tools import knowledge_base, github_api
from llm_client import get_ai_analysis_and_plan
import os

# --- PAGE CONFIGURATION (No changes) ---
st.set_page_config(
    page_title="AI Knowledge Orchestrator",
    page_icon="🧠",
    layout="wide"
)

# --- HELPER & INITIALIZATION (No changes) ---
def parse_llm_response(response: str):
    if not response or not isinstance(response, str):
        return "Invalid response from LLM (empty or not a string).", []

    try:
        cause_match = re.search(r"### Root Cause Analysis\s*(.*?)\s*(?:### Solutions|### Generate Solutions)", response, re.DOTALL | re.IGNORECASE)
        solutions_text_match = re.search(r"### (?:Solutions|### Generate Solutions)\s*(.*)", response, re.DOTALL | re.IGNORECASE)
        
        if cause_match:
            root_cause_analysis = cause_match.group(1).strip()
        elif solutions_text_match:
            fallback_analysis = response.split(solutions_text_match.group(0))[0].strip()
            root_cause_analysis = fallback_analysis if fallback_analysis else "AI did not provide a distinct root cause analysis."
        else:
            root_cause_analysis = "AI did not provide a root cause analysis."

        if not solutions_text_match:
             raise ValueError("The '### Solutions' or '### Generate Solutions' header was not found.")
        
        solutions_full_text = solutions_text_match.group(1)
        solution_blocks = re.split(r"(?:\#\#\#\#|\*\*)?\s*Solution \d+:\s*(?:\*\*)?", solutions_full_text, flags=re.IGNORECASE)

        solutions = []
        for block in solution_blocks:
            if not block.strip():
                continue

            title = block.splitlines()[0].strip() if block.splitlines() else "Untitled Solution"
            description = re.search(r"\*\*Description:\*\*\s*(.*?)\s*(?:\*\*Pros:\*\*|\*\*Cons:\*\*|\*\*Confidence Score:\*\*|\*\*Action:\*\*)", block, re.DOTALL | re.IGNORECASE)
            pros = re.search(r"\*\*Pros:\*\*\s*(.*?)\s*(?:\*\*Cons:\*\*|\*\*Confidence Score:\*\*|\*\*Action:\*\*)", block, re.DOTALL | re.IGNORECASE)
            cons = re.search(r"\*\*Cons:\*\*\s*(.*?)\s*(?:\*\*Confidence Score:\*\*|\*\*Action:\*\*)", block, re.DOTALL | re.IGNORECASE)
            confidence = re.search(r"\*\*?Confidence Score:\*\*?\s*(\d+)\s*%?", block, re.DOTALL | re.IGNORECASE)
            action_json_match = re.search(r"```json\s*(\{.*?\})\s*```", block, re.DOTALL)

            current_solution = {'title': title}
            if description: current_solution['description'] = description.group(1).strip()
            if pros: current_solution['pros'] = pros.group(1).strip()
            if cons: current_solution['cons'] = cons.group(1).strip()
            if confidence: current_solution['confidence'] = int(confidence.group(1))
            
            if action_json_match:
                try:
                    json_str = action_json_match.group(1).replace("`", "")
                    current_solution['action'] = json.loads(json_str)
                except json.JSONDecodeError:
                    current_solution['action'] = {"error": "Invalid JSON in response"}
            
            if 'description' in current_solution or 'action' in current_solution:
                solutions.append(current_solution)

        if not solutions:
            raise ValueError("Found the 'Solutions' section, but could not parse any individual solutions from it.")

        return root_cause_analysis, solutions
    except Exception as e:
        return f"Could not parse analysis due to a structural error: {e}. See raw response below.", []

if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
    st.session_state.root_cause = ""
    st.session_state.solutions = []
    st.session_state.context = ""
    st.session_state.raw_response = ""
    st.session_state.trigger = ""
    st.session_state.repo_link = ""
    st.session_state.assignee = ""

# --- UI LAYOUT (No changes) ---
st.title("🧠 AI Knowledge Orchestrator Agent")
st.markdown("Moves beyond a 'Chatbot' to AI Research + Orchestration + Consensus for IT teams. Powered by `Groq`.")
st.divider()

st.subheader("Step 1: The Trigger")
st.markdown("Enter the problem, alert, or ticket description that activates the agent.")
default_trigger = "High latency detected on the user authentication service after the latest deployment. Users are reporting 500 errors during login."
trigger_text = st.text_area("Trigger Description", value=default_trigger, height=100)
repo_link = st.text_input("GitHub Repository Link", placeholder="e.g., https://github.com/user/repo")
assignee = st.text_input(
    "Assignee's GitHub Username", 
    placeholder="e.g., github_username",
    help="This user must be a collaborator on the repository to be assigned."
)

if st.button("🚀 Analyze Issue", type="primary"):
    st.session_state.analysis_complete = False
    st.session_state.solutions = []
    if not trigger_text or not repo_link or not assignee:
        st.warning("Please enter a trigger description, repository link, and assignee.")
    else:
        st.session_state.trigger = trigger_text
        st.session_state.repo_link = repo_link
        st.session_state.assignee = assignee
        with st.spinner("Step 2: Gathering Context from GitHub..."):
            code_context = knowledge_base.get_code_context(trigger_text, repo_link)
            docs_context = knowledge_base.get_docs_context(trigger_text)
            st.session_state.context = f"**Code Context from {repo_link}:**\n{code_context}\n\n**Documentation Context:**\n{docs_context}"
        
        with st.spinner("Step 3: AI Brain (Groq Llama3) is analyzing..."):
            llm_response = get_ai_analysis_and_plan(trigger_text, st.session_state.context)
            st.session_state.raw_response = llm_response
            root_cause, solutions = parse_llm_response(llm_response)
            st.session_state.root_cause = root_cause
            st.session_state.solutions = solutions
            st.session_state.analysis_complete = True
            st.rerun()

# --- RENDER RESULTS ---
if st.session_state.analysis_complete:
    st.divider()
    with st.expander("View Gathered Context"):
        st.markdown(st.session_state.context)
    st.subheader("🤖 DevBrain Analysis")
    st.info(f"**Root Cause Analysis:** {st.session_state.root_cause}")
    st.divider()
    st.subheader("Step 4: Consensus - Proposed Solutions")
    
    if not st.session_state.solutions:
        st.error("No valid solutions were generated or parsed from the AI's response.")
        if st.session_state.raw_response:
            st.subheader("Raw AI Response (for debugging):")
            st.text_area("LLM Output", st.session_state.raw_response, height=300)
    else:
        for i, sol in enumerate(st.session_state.solutions):
            with st.container(border=True):
                st.markdown(f"**{sol.get('title', f'Solution {i+1}')}**")
                st.markdown(sol.get('description', 'No description provided.'))
                
                col1, col2, col3 = st.columns(3)

                with col1: st.success(f"**Pros:** {sol.get('pros', 'N/A')}")
                with col2: st.warning(f"**Cons:** {sol.get('cons', 'N/A')}")
                with col3: st.metric("Confidence Score", f"{sol.get('confidence', 0)}%")

                if st.button(f"✅ Execute Solution {i+1}", key=f"execute_{i}"):
                    action = sol.get('action', {})
                    tool = action.get('tool')
                    function = action.get('function')
                    params = action.get('params', {})
                    st.subheader("Step 5: Orchestrating Action")
                    
                    with st.spinner("Executing action..."):
                        result = ""
                        if tool == 'github' and function:
                            repo_name = st.session_state.repo_link.replace("https://github.com/", "").strip("/")
                            
                            if function == 'trigger_workflow':
                                result = github_api.trigger_workflow(
                                    workflow_name=params.get('workflow_name', 'N/A'),
                                    parameters=params.get('parameters', {})
                                )
                            elif function == 'create_github_issue':
                                result = github_api.create_github_issue(
                                    repo=repo_name,
                                    title=params.get('title', 'AI-Generated Issue'),
                                    body=params.get('body', 'No description provided by AI.'),
                                    assignee=st.session_state.assignee
                                )
                            else:
                                result = f"Error: Unknown function '{function}' for tool 'github'."
                        else:
                            result = f"Error: Could not execute. Action data from AI was incomplete or invalid.\nReceived: {json.dumps(action, indent=2)}"
                        
                        # === START OF UPDATED CODE ===
                        # Better display logic for the result
                        if "Error" in result:
                            st.error(result)
                        else:
                            st.success("Action Orchestrated Successfully!")
                            # Use markdown for results that contain links, otherwise use a code block
                            if "http" in result:
                                st.markdown(result)
                            else:
                                st.code(result, language="text")
                        # === END OF UPDATED CODE ===
                        
                        st.info("Step 6: Outcome recorded to knowledge base.")

if st.session_state.analysis_complete:
    if st.button("Reset and Start New Analysis"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()