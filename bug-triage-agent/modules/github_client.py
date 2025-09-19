from github import Github
import streamlit as st

def get_github_issues(repo_name):
    """
    Fetches open issues from a given GitHub repository.
    """
    try:
        g = Github(st.secrets["GITHUB_TOKEN"])
        repo = g.get_repo(repo_name)
        # We fetch only one issue to provide a detailed analysis for it
        issues = repo.get_issues(state='open')
        return list(issues)
    except Exception as e:
        st.error(f"Failed to fetch issues from GitHub: {e}")
        return []