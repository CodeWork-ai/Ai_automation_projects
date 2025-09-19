# tools/knowledge_base.py
import os
import requests
from dotenv import load_dotenv

load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

def get_code_context(issue_description: str, repo_link: str) -> str:
    """Simulates searching the codebase for relevant context."""
    print("...Searching codebase...")
    
    if not GITHUB_TOKEN:
        return "Error: GITHUB_TOKEN not found. Cannot search repository."

    repo_name = repo_link.split("github.com/")[-1]
    url = f"https://api.github.com/repos/{repo_name}/search/code?q={issue_description.split()[0]}"
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json",
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        results = response.json()
        if results.get("items"):
            return f"Found relevant code in {results['items'][0]['path']}"
        return "No specific code context found for the issue."
    except requests.exceptions.RequestException as e:
        return f"Could not search repository: {e}"


def get_docs_context(issue_description: str) -> str:
    """Simulates searching internal documentation (e.g., Confluence)."""
    print("...Searching internal documentation...")
    if "auth" in issue_description.lower():
        return """
        - Document: 'Authentication Service Architecture'
        - Key Point: The service uses a primary database and a read-replica.
        - On-call Procedure: For high latency, first check replica lag, then database connection pool size.
        """
    return "No relevant documentation found."