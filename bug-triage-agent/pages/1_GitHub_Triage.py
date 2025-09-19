import streamlit as st
from modules.github_client import get_github_issues
from modules.triage_agent import triage_issue

st.set_page_config(page_title="GitHub Triage", page_icon="assets/icon.png")

st.title("GitHub Issue Triage Agent")

repo_name = st.text_input("Enter GitHub Repository (e.g., 'owner/repo'):")

if st.button("Fetch and Triage Issues"):
    if repo_name:
        with st.spinner("Fetching issues from GitHub..."):
            issues = get_github_issues(repo_name)

        if issues:
            st.success(f"Found {len(list(issues))} open issues.")
            for issue in issues:
                st.write(f"### [{issue.title}]({issue.html_url})")
                st.write(f"**Issue Number:** #{issue.number}")
                st.write(f"**Author:** {issue.user.login}")

                with st.spinner(f"Triaging issue #{issue.number}..."):
                    triage_result = triage_issue(issue.title, issue.body)
                    st.info(triage_result)
        else:
            st.warning("No open issues found in the repository.")
    else:
        st.error("Please enter a repository name.")