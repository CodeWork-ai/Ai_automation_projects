import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

def trigger_workflow(workflow_name: str, parameters: dict) -> str:
    """Simulates triggering a GitHub Actions workflow and returns a summary."""
    # This function is now complete and requires no further changes.
    summary = f"""
    ```text
    ACTION: Trigger GitHub Workflow (SIMULATED)
    -------------------------------------------
    Workflow Name: '{workflow_name}'
    Parameters: {json.dumps(parameters, indent=2)}
    -------------------------------------------
    This is a simulation. In a real environment, this would trigger the specified
    GitHub Actions workflow file (e.g., '{workflow_name}.yml').
    ```
    """
    return summary


def create_github_issue(repo: str, title: str, body: str, assignee: str) -> str:
    """Creates an issue in a GitHub repository and returns a link."""
    if not GITHUB_TOKEN:
        return "Error: GITHUB_TOKEN not found in .env file. Please set it."

    url = f"https://api.github.com/repos/{repo}/issues"
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json",
    }
    data = {
        "title": title,
        "body": body,
        "assignees": [assignee]
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        
        if response.status_code == 422:
            # Error handling for non-collaborator assignment
            error_details = response.json()
            error_message = error_details.get('message', 'Validation failed.')
            specific_error = f"The user '{assignee}' is likely not a collaborator on the '{repo}' repository. Please invite them first."
            return (f"Error: GitHub API returned a 422 Unprocessable Entity.\n"
                    f"Message: {error_message}\n"
                    f"**Likely Cause:** {specific_error}")
        
        response.raise_for_status()
        
        # === START OF UPDATED CODE ===
        # Extract the URL from the successful API response and format it with Markdown
        issue_url = response.json().get("html_url")
        return f"Successfully created issue in **{repo}** and assigned to **{assignee}**.\n\n➡️ [**Click here to view the new issue**]({issue_url})"
        # === END OF UPDATED CODE ===

    except requests.exceptions.RequestException as e:
        return f"Error connecting to GitHub API: {e}"