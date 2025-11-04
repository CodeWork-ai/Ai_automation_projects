# Import the functions from our other modules
from pubmed_client import search_pubmed, fetch_abstracts
from summarizer import summarize_text

def run_agent(query, max_articles=3):
    """
    Runs the complete AI agent process: search, fetch, and summarize.
    
    Args:
        query (str): The medical topic to research.
        max_articles (int): The number of articles to process.
    """
    print("-" * 50)
    print(f"Starting AI Agent for query: '{query}'")
    print("-" * 50)

    # Step 1: Search PubMed for article IDs
    article_ids = search_pubmed(query, max_results=max_articles)

    if not article_ids:
        print("Agent run finished: No articles were found.")
        return

    # Step 2: Fetch the abstracts for these articles
    abstracts = fetch_abstracts(article_ids)
    
    if not abstracts:
        print("Agent run finished: Could not fetch abstracts.")
        return
        
    print("\n" + "=" * 50)
    print("SUMMARIZATION RESULTS")
    print("=" * 50 + "\n")

    # Step 3: Summarize each abstract
    for pmid, abstract in abstracts.items():
        print(f"--- Article PMID: {pmid} ---")
        print(f"Original Abstract: {abstract[:300]}...") # Print a snippet of the abstract
        
        print("\nGenerating summary...")
        summary = summarize_text(abstract)
        
        print(f"\n✅ Plain-Language Summary:")
        print(summary)
        print("-" * 50 + "\n")

if __name__ == "__main__":
    # You can change this query to whatever you want to research
    user_query = "latest research on CAR-T therapy for lymphoma"
    run_agent(user_query)