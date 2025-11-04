import requests
import xml.etree.ElementTree as ET

def search_pubmed(query, max_results=5):
    """
    Searches PubMed for a given query and returns a list of article IDs.
    
    Args:
        query (str): The search term (e.g., "treatment for diabetes").
        max_results (int): The maximum number of results to return.

    Returns:
        list: A list of PubMed article IDs (PMIDs) as strings.
    """
    print(f"Searching PubMed for: '{query}'...")
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "sort": "relevance"
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()  # Raises an HTTPError for bad responses (4XX or 5XX)
        
        root = ET.fromstring(response.content)
        id_list = [id_elem.text for id_elem in root.findall(".//Id")]
        
        if not id_list:
            print("No articles found for the query.")
            return []
            
        print(f"Found {len(id_list)} article IDs.")
        return id_list
        
    except requests.exceptions.RequestException as e:
        print(f"An error occurred during PubMed search: {e}")
        return []

def fetch_abstracts(id_list):
    """
    Fetches the abstracts for a list of PubMed article IDs.

    Args:
        id_list (list): A list of PubMed article IDs (PMIDs).

    Returns:
        dict: A dictionary mapping PMID to its abstract text.
    """
    if not id_list:
        return {}
        
    print("Fetching abstracts for the found articles...")
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    ids = ",".join(id_list)
    params = {
        "db": "pubmed",
        "id": ids,
        "retmode": "xml",
        "rettype": "abstract"
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        
        abstracts = {}
        root = ET.fromstring(response.content)
        
        for article in root.findall(".//PubmedArticle"):
            pmid_element = article.find(".//PMID")
            if pmid_element is None:
                continue
            
            pmid = pmid_element.text
            abstract_element = article.find(".//Abstract/AbstractText")
            
            if abstract_element is not None and abstract_element.text:
                abstracts[pmid] = abstract_element.text.strip()
            else:
                abstracts[pmid] = "No abstract available for this article."
                
        print(f"Successfully fetched {len(abstracts)} abstracts.")
        return abstracts
        
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while fetching abstracts: {e}")
        return {}