import keepa
from config import KEEPA_API_KEY

def get_price_history(asin: str, domain_code: str) -> dict:
    """
    Fetches price history for a given Amazon product ASIN using the Keepa API.
    
    Args:
        asin (str): The Amazon Standard Identification Number of the product.
        domain_code (str): The two-letter country code (e.g., 'in', 'com').

    Returns:
        dict: A dictionary containing key price statistics.
    """
    if not KEEPA_API_KEY:
        print("Warning: KEEPA_API_KEY is not set. Cannot fetch price history.")
        return {}

    try:
        api = keepa.Keepa(KEEPA_API_KEY)
        # Keepa uses integer IDs for domains. This map translates them.
        domain_map = {'com': 1, 'in': 10, 'co.uk': 3, 'de': 4, 'fr': 5, 'ca': 7, 'it': 9, 'es': 11}
        keepa_domain_id = domain_map.get(domain_code, 1) # Default to .com

        # Query Keepa for the product, requesting 90-day statistics
        products = api.query(asin, domain=keepa_domain_id, stats=90)
        
        if not products:
            return {}

        product = products[0]
        stats = product['stats']
        
        # Keepa returns prices in the lowest denomination (cents/paisa). Divide by 100.
        current_price = stats['current'][0] / 100 if stats['current'][0] > 0 else 'N/A'
        avg_90_days = stats['avg'][0] / 100 if stats['avg'][0] > 0 else 'N/A'
        lowest_price = stats['min'][0] / 100 if stats['min'][0] > 0 else 'N/A'
        highest_price = stats['max'][0] / 100 if stats['max'][0] > 0 else 'N/A'

        history = {
            "current_price_numeric": current_price,
            "average_90_days_price": avg_90_days,
            "lowest_price": lowest_price,
            "highest_price": highest_price
        }
        print(f"Successfully fetched price history for ASIN {asin}: {history}")
        return history

    except Exception as e:
        print(f"An error occurred with Keepa API for ASIN {asin}: {e}")
        return {}