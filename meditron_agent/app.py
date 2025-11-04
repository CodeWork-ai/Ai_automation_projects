import streamlit as st
from pubmed_client import search_pubmed, fetch_abstracts
from summarizer import generate_response, model_is_available
import re

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="MEDITRON AI Research Assistant", page_icon="🤖", layout="wide")

# --- CONSOLIDATION FUNCTION ---
def generate_structured_summary(abstracts_dict):
    if not abstracts_dict: return "No content to summarize.", []

    st.write("➡️ Step 1: Combining abstracts for the Groq API prompt...")
    combined_abstracts = "\n\n---\n\n".join(abstracts_dict.values())
    pmid_list = list(abstracts_dict.keys())
    
    # --- UPDATED TEXT FOR USER ---
    st.write("➡️ Step 2: Sending request to Groq's Llama 3.1 model...")
    
    prompt_content = f"""
    You are an expert medical research analyst. Your task is to synthesize the key findings from a collection of research abstracts into a clear, structured summary.

    Here are the abstracts:
    ---
    {combined_abstracts}
    ---

    Based ONLY on the text from the abstracts provided above, generate a structured summary. The summary must have the following format exactly:
    **Overall Finding:** [A single, impactful sentence that summarizes the main conclusion.]
    **Key Points:** 
    - [A bullet point summarizing the first key detail.]
    - [A bullet point summarizing the second key detail.]
    - [A bullet point summarizing the third key detail.]
    **Conclusion:** [A brief concluding sentence about the potential impact.]
    """
    
    messages = [{"role": "user", "content": prompt_content}]
    
    structured_summary = generate_response(messages)
    
    return structured_summary, pmid_list

# --- APP LAYOUT ---
st.title("🚀 MEDITRON AI Research Assistant (Powered by Groq)")

if not model_is_available:
    st.error(
        "CRITICAL ERROR: Groq API client failed to initialize. "
        "Please make sure you have created a `.env` file with your `GROQ_API_KEY`."
    )
else:
    st.markdown("### Get an ultra-fast, structured summary from the latest medical research.")
    st.markdown("---")

    user_query = st.text_input("Enter a medical topic to research:", placeholder="e.g., 'CAR-T therapy for lymphoma'")
    num_articles = st.slider("Number of articles to analyze:", min_value=2, max_value=10, value=3)

    if st.button('Generate Structured Summary', type="primary"):
        if not user_query:
            st.error("Please enter a topic to research.")
        else:
            with st.spinner(f"Searching PubMed for '{user_query}'..."):
                article_ids = search_pubmed(user_query, max_results=num_articles)
            
            if not article_ids:
                st.warning("No articles found. Please try a different topic.")
            else:
                with st.spinner(f"Found {len(article_ids)} articles. Fetching content..."):
                    abstracts = fetch_abstracts(article_ids)
                st.success(f"Successfully fetched {len(abstracts)} articles.")
                st.markdown("---")

                st.header("🔬 Consolidated & Structured Findings")
                with st.spinner("Sending request to Groq's high-speed AI..."):
                    final_summary, source_pmids = generate_structured_summary(abstracts)
                
                st.markdown(final_summary)
                
                st.markdown("---")
                st.subheader("Source Articles Analyzed")
                source_links = [f"[{pmid}](https://pubmed.ncbi.nlm.nih.gov/{pmid}/)" for pmid in source_pmids]
                st.markdown(", ".join(source_links), unsafe_allow_html=True)
                
                with st.expander("Show Original Abstracts"):
                    for pmid, abstract in abstracts.items():
                        st.subheader(f"📄 Abstract for Article PMID: {pmid}")
                        st.write(abstract)