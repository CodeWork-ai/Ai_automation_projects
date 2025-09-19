import streamlit as st
from modules.github_client import get_github_issues
from modules.triage_agent import get_detailed_analysis
import re

# --- Page Configuration ---
st.set_page_config(
    page_title="GitHub Bug Triage Agent",
    page_icon="assets/icon.png",
    layout="wide"
)

# --- MORE ROBUST HELPER FUNCTION TO PARSE THE AI'S RESPONSE ---
def parse_ai_summary(markdown_text):
    """
    Parses the Markdown response from the AI to extract key triage details.
    This version is more robust and handles different formatting (bold vs. lists).
    """
    priority, category, suggestion = "N/A", "N/A", "Could not be determined."

    # Search for patterns, ignoring case for flexibility
    priority_match = re.search(r"(\*\*Priority:\*\*|\* Priority:)\s*(.*)", markdown_text, re.IGNORECASE)
    category_match = re.search(r"(\*\*Category:\*\*|\* Category:)\s*(.*)", markdown_text, re.IGNORECASE)
    suggestion_match = re.search(r"(\*\*Actionable Title:\*\*|\* Actionable Title:)\s*(.*)", markdown_text, re.IGNORECASE)

    if priority_match:
        priority = priority_match.group(2).strip()
    if category_match:
        category = category_match.group(2).strip()
    if suggestion_match:
        suggestion = suggestion_match.group(2).strip()
        
    return priority, category, suggestion

# --- Initialize Session State ---
# This helps us store information across user interactions
if 'issues_list' not in st.session_state:
    st.session_state.issues_list = None
if 'analysis_cache' not in st.session_state:
    st.session_state.analysis_cache = {}

# --- Page Title and Description ---
st.title("🤖 AI-Powered GitHub Bug Triage Agent")
st.markdown(
    """
    **Step 1:** Enter a public GitHub repository and click "Fetch Issues" to see all open issues.
    **Step 2:** Select an issue from the list to get a detailed, AI-powered analysis.
    """
)

# --- Input Section ---
st.subheader("Enter GitHub Repository")
repo_name = st.text_input(
    "Enter a public GitHub Repository (e.g., `owner/repo`):",
    placeholder="streamlit/streamlit"
)

# --- NEW WORKFLOW: Step 1 - Fetching Issues ---
if st.button("Fetch All Open Issues"):
    # Clear any previous results
    st.session_state.issues_list = None
    st.session_state.analysis_cache = {}
    
    if repo_name:
        with st.spinner(f"Fetching open issues from `{repo_name}`..."):
            issues = get_github_issues(repo_name)
            if issues:
                st.session_state.issues_list = issues
                st.success(f"Found {len(issues)} open issues. Please select an issue below to analyze.")
            else:
                st.warning("No open issues found in this repository.")
    else:
        st.error("Please enter a repository name.")

# --- NEW WORKFLOW: Step 2 - Displaying the List for Selection ---
if st.session_state.issues_list:
    st.markdown("---")
    st.subheader("Select an Issue to Analyze")
    
    # We will display the analysis for one selected issue
    selected_issue = None
    
    # Display issues in columns for a neat UI
    num_issues = len(st.session_state.issues_list)
    cols = st.columns(3 if num_issues > 2 else num_issues or 1)
    for i, issue in enumerate(st.session_state.issues_list):
        if cols[i % 3].button(f"#{issue.number}: {issue.title}", key=issue.id):
            selected_issue = issue
    
    # --- Analysis Section for the Selected Issue ---
    if selected_issue:
        # Check if we have analyzed this issue before (caching)
        if selected_issue.id in st.session_state.analysis_cache:
            triage_result = st.session_state.analysis_cache[selected_issue.id]
        else:
            # If not, call the AI agent
            with st.spinner(f"AI agent is analyzing issue #{selected_issue.number}... This may take a moment."):
                triage_result = get_detailed_analysis(selected_issue.title, selected_issue.body)
                # Save the result to cache to avoid re-running
                st.session_state.analysis_cache[selected_issue.id] = triage_result
        
        # Now parse the result (either from cache or new) and display everything at once
        priority, category, suggestion = parse_ai_summary(triage_result)
        
        st.markdown("---")
        st.header(f"Analysis for Issue #{selected_issue.number}")
        
        with st.expander("View Full AI Analysis", expanded=True):
            st.markdown(f"*[Link to Issue on GitHub]({selected_issue.html_url})*")
            
            # Display the rich, correctly parsed summary at the top
            st.subheader(f"📌 Priority: {priority} | 🏷️ Category: {category}")
            st.caption(f"Suggested Action: {suggestion}")
            st.markdown("---")
            
            # Display the full detailed report from the AI
            st.markdown(triage_result)