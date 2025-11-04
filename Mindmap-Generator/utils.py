# utils.py

import requests
from bs4 import BeautifulSoup
from groq import Groq
import streamlit as st
import datetime
import time

@st.cache_data(show_spinner=False)
def search_academic_papers(topic, api_key):
    """
    Searches for academic papers, prioritizing the highest citation count.
    """
    max_retries = 3
    base_delay = 5
    headers = {"x-api-key": api_key}

    for attempt in range(max_retries):
        try:
            current_year = datetime.date.today().year
            start_year = current_year - 4
            year_range = f"{start_year}-{current_year}"

            response = requests.get(
                "https://api.semanticscholar.org/graph/v1/paper/search",
                params={ "query": topic, "limit": 20, "fields": "title,authors,year,abstract,url,citationCount", "year": year_range },
                headers=headers
            )
            response.raise_for_status()
            papers = response.json().get("data", [])

            papers_to_sort = [p for p in papers if p.get('citationCount') is not None and p.get('year') is not None]

            sorted_papers = sorted(
                papers_to_sort,
                key=lambda p: p['citationCount'],
                reverse=True
            )

            return sorted_papers

        except requests.exceptions.HTTPError as http_err:
            if http_err.response.status_code == 429:
                if attempt < max_retries - 1:
                    delay = base_delay * (attempt + 1)
                    st.warning(f"Rate limit hit. Automatically retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    st.error("API is still busy after multiple retries. Please try again in a few minutes.")
                    return None
            else:
                st.error(f"An HTTP error occurred: {http_err}")
                return None
        except requests.exceptions.RequestException as e:
            st.error(f"A network error occurred: {e}")
            return None
    return None

@st.cache_data(show_spinner=False)
def get_paper_content(url, scraper_api_key):
    """ Standard scraping attempt with JavaScript rendering. """
    try:
        payload = {'api_key': scraper_api_key, 'url': url, 'render': 'true'}
        response = requests.get('http://api.scraperapi.com', params=payload)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = [p.get_text() for p in soup.find_all('p')]
        return "\n".join(paragraphs).strip() if paragraphs else ""
    except requests.exceptions.RequestException:
        return ""

@st.cache_data(show_spinner=False)
def get_premium_paper_content(url, scraper_api_key):
    """ Premium scraping attempt using different settings as a fallback. """
    try:
        payload = {'api_key': scraper_api_key, 'url': url, 'render': 'true', 'country_code': 'us'}
        response = requests.get('http://api.scraperapi.com', params=payload, timeout=60)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = [p.get_text() for p in soup.find_all('p')]
        return "\n".join(paragraphs).strip() if paragraphs else ""
    except requests.exceptions.RequestException:
        return ""

# This function remains as it is the source for the workflow generation
def deep_analysis(text, paper, groq_api_key):
    """ Performs a fine-grained, structured analysis with verbatim data extraction and identifies missing information. """
    url = paper.get('url', 'URL not available')
    try:
        client = Groq(api_key=groq_api_key)
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert academic analysis bot. Your primary function is to extract verifiable data points from research papers and to identify which key pieces of information are missing. You must adhere strictly to the requested output format."
                },
                {
                    "role": "user",
                    "content": f"""Please perform a two-part analysis of the following research paper text.

**Part 1: Detailed Analysis**
Go through the text and extract the information for the sections listed below. It is crucial that you **only include a section in your output if you can find relevant data for it in the text**. If no information can be found for an entire section (e.g., "Dataset Details"), omit that section completely from your response. Present the extracted information as a well-structured Markdown document.

- **0. Source URL**: {url}
- **1. Core Hypothesis**: What is the exact research question or statement the paper investigates?
- **2. Detailed Methodology**: Describe the key methods, techniques, and experimental workflow.
- **3. Dataset Details**: Name, Source/URL, Size & Characteristics.
- **4. Model & Architecture**: Model Name, Architecture Details.
- **5. Implementation Details**: Data Preprocessing, Training Configuration, Evaluation Protocol.
- **6. Quantitative Performance Metrics**: Create a Markdown table of all reported metrics.
- **7. Qualitative Findings**: List the key non-numerical insights or discoveries.
- **8. Conclusion & Significance**: What is the paper's final conclusion and its stated impact?
- **9. Code Availability**: Does the paper mention a public repository (e.g., GitHub)?

---
**Part 2: Suggestions for Deeper Investigation**
After the detailed analysis, add a final section titled `### 🤔 Suggestions for Deeper Investigation`. In this section, create a bulleted list of important, specific details that were **not found** in the provided text.

---
**Full Paper Text to Analyze:**
---
{text}
---"""
                }
            ],
            model="openai/gpt-oss-20b",
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        st.error(f"Error during deep analysis: {e}")
        return ""


def generate_project_workflow(analysis_text, groq_api_key):
    """
    Converts a detailed paper analysis into an actionable project workflow,
    suggesting improvements and modern tooling.
    """
    try:
        client = Groq(api_key=groq_api_key)
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a senior AI research scientist and project manager. Your task is to convert a research paper's analysis "
                        "into a detailed, actionable, end-to-end project plan. You must identify gaps in the paper's methodology and "
                        "propose concrete improvements using modern best practices and tools. Format your response in clear, structured Markdown."
                    )
                },
                {
                    "role": "user",
                    "content": f"""
Analyze the structured information from the research paper below. Based on this, create a complete, step-by-step project workflow that a research team could use to replicate and enhance the original study.

**Your Task:**
1.  **Structure the Workflow:** Organize the plan into logical phases (e.g., Setup, Data Curation, Modeling, Evaluation, etc.).
2.  **Identify and Improve:** Scrutinize the paper's methodology. Where details are missing or methods are outdated, you **must** propose a specific, modern alternative.
3.  **Highlight Enhancements:** Clearly label your suggestions in the workflow using the format `[Improvement: Your suggestion here]`. For example, `[Improvement: Use MLflow to track all experiment runs, parameters, and metrics for better reproducibility.]`.
4.  **Be Specific:** Provide concrete steps. Instead of "preprocess the data," suggest specific techniques mentioned or appropriate for the context.

**Example of a good workflow step:**
"**Phase 2: Data Preprocessing & Augmentation**
-   **Image Normalization:** Standardize pixel values across the dataset. The paper did not specify the method.
    -   `[Improvement: Normalize images using the ImageNet mean and standard deviation, as it is a common practice for transfer learning models like Xception.]`
-   **Data Augmentation:** The paper does not mention augmentation.
    -   `[Improvement: Introduce data augmentation (random rotations, flips, brightness adjustments) using a library like Albumentations to prevent overfitting and improve model generalization.]`"

**Analysis of the Research Paper to Convert:**
---
{analysis_text}
---
"""
                }
            ],
            model="openai/gpt-oss-20b",  # Using a more powerful model for better reasoning
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        st.error(f"Error generating project workflow: {e}")
        return ""