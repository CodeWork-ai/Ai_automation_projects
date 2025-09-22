from analyzer import ContentAnalyzer

class DigestGenerator:
    def __init__(self, model_name="t5-small"):
        """Initialize the digest generator with a content analyzer"""
        self.analyzer = ContentAnalyzer(model_name)
    
    def generate_digest(self, scraped_data, historical_data=None, threshold=0.7):
        """Generate market research digest from scraped data"""
        print("Generating market research digest...")
        
        # Generate the digest using the analyzer
        digest = self.analyzer.generate_market_digest(
            scraped_data=scraped_data,
            historical_data=historical_data,
            threshold=threshold
        )
        
        print("Digest generated successfully!")
        return digest
    
    def save_digest(self, digest, filename="market_research_digest.txt"):
        """Save the digest to a file"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(digest)
            print(f"Digest saved to {filename}")
            return True
        except Exception as e:
            print(f"Error saving digest: {e}")
            return False
    
    def print_digest(self, digest):
        """Print the digest to console"""
        print("\n" + "="*70)
        print("MARKET RESEARCH DIGEST")
        print("="*70)
        print(digest)
        print("="*70)