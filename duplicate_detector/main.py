import os
from api_clients import get_groq_client, get_firecrawl_client, get_google_search_service
from plagiarism_checker import (
    chunk_text,
    search_web_for_chunk,
    scrape_url_content,
    compare_with_llm,
    crawl_website,
    read_local_directory,
    find_internal_duplicates,
    generate_optimization_report
)

def main():
    """Main function to run the Duplicate Content Detector Agent."""
    print("--- Duplicate Content Detector Agent ---")

    # --- Initialize API Clients ---
    groq_client = get_groq_client()
    firecrawl_client = get_firecrawl_client()
    search_service = get_google_search_service()
    google_cse_id = os.environ.get("GOOGLE_CSE_ID")

    # --- Get User Input ---
    print("\nSelect an option:")
    print("1. Check a block of text")
    print("2. Scan a website")
    print("3. Scan a local directory")
    choice = input("Enter your choice (1, 2, or 3): ")

    content_map = {}
    if choice == '1':
        user_content = input("Please paste the content you want to check:\n")
        content_map['user_input'] = user_content
    elif choice == '2':
        url = input("Enter the website URL to scan: ")
        content_map = crawl_website(url, firecrawl_client)
    elif choice == '3':
        path = input("Enter the local directory path to scan: ")
        content_map = read_local_directory(path)
    else:
        print("Invalid choice.")
        return

    if not content_map:
        print("No content found to analyze.")
        return

    print(f"\nFound {len(content_map)} document(s) to analyze.")

    # --- Run Internal Analysis ---
    print("\n[Step 1/3] Scanning for internal duplicates...")
    internal_duplicates = find_internal_duplicates(content_map)
    if internal_duplicates:
        print("Found potential internal duplicates:")
        for doc, duplicates in internal_duplicates.items():
            print(f"- '{os.path.basename(doc)}' is similar to:")
            for dup in duplicates:
                print(f"  - '{os.path.basename(dup['similar_to'])}' (Score: {dup['score']:.2f})")
    else:
        print("No significant internal duplicates found.")

    # --- Run External Plagiarism and Optimization Analysis ---
    print("\n[Step 2/3] Scanning for external plagiarism...")
    final_reports = {}
    for doc_name, content in content_map.items():
        print(f"\nAnalyzing '{os.path.basename(doc_name)}' for external plagiarism...")
        chunks = chunk_text(content)
        if not chunks:
            print("  -> Document is empty, skipping external search.")
            continue
        
        urls = search_web_for_chunk(chunks[0], search_service, google_cse_id)
        
        plagiarism_results = []
        if urls:
            scraped_content = scrape_url_content(urls[0], firecrawl_client)
            if scraped_content:
                analysis = compare_with_llm(content, scraped_content, urls[0], groq_client)
                plagiarism_results.append(analysis)

        # --- Generate Final Report ---
        print(f"Generating optimization report for '{os.path.basename(doc_name)}'...")
        final_reports[doc_name] = generate_optimization_report(content, plagiarism_results, groq_client)

    # --- Display All Reports ---
    print("\n[Step 3/3] Final Analysis and Optimization Reports ---")
    for doc_name, report in final_reports.items():
        print(f"\n--- Report for: {os.path.basename(doc_name)} ---")
        print(report)
        print("-" * 40)

if __name__ == "__main__":
    main()
