import pandas as pd
import json
import numpy as np
from tqdm import tqdm

class DataLoader:
    """Load Amazon Electronics dataset (JSONL format)"""
    
    def __init__(self, ratings_path, metadata_path):
        self.ratings_path = ratings_path
        self.metadata_path = metadata_path
    
    def load_ratings(self, sample_size=None):
        """Load ratings from JSONL file (Electronics.jsonl)"""
        print("Loading ratings from Electronics.jsonl...")
        
        ratings = []
        
        with open(self.ratings_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Reading ratings"):
                try:
                    review = json.loads(line.strip())
                    
                    # Extract fields from the review
                    ratings.append({
                        'user_id': str(review.get('user_id', review.get('reviewerID', ''))),
                        'product_id': str(review.get('parent_asin', review.get('asin', ''))),
                        'rating': float(review.get('rating', review.get('overall', 0))),
                        'timestamp': review.get('timestamp', review.get('unixReviewTime', 0))
                    })
                except Exception as e:
                    continue
        
        df = pd.DataFrame(ratings)
        
        # Clean data
        df = df.dropna(subset=['user_id', 'product_id', 'rating'])
        df = df[df['rating'] > 0]
        df['user_id'] = df['user_id'].astype(str)
        df['product_id'] = df['product_id'].astype(str)
        
        # Remove empty IDs
        df = df[(df['user_id'] != '') & (df['product_id'] != '')]
        
        if sample_size and sample_size < len(df):
            df = df.sample(n=sample_size, random_state=42)
        
        print(f"✓ Loaded {len(df):,} ratings")
        print(f"  - Unique users: {df['user_id'].nunique():,}")
        print(f"  - Unique products: {df['product_id'].nunique():,}")
        
        return df
    
    def load_metadata(self):
        """Load product metadata from JSONL file (meta_Electronics.jsonl)"""
        print("Loading product metadata from meta_Electronics.jsonl...")
        
        products = []
        product_ids_seen = set()
        
        with open(self.metadata_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Reading metadata"):
                try:
                    product = json.loads(line.strip())
                    
                    # Get product ID (can be 'parent_asin' or 'asin')
                    prod_id = str(product.get('parent_asin', product.get('asin', '')))
                    
                    if not prod_id or prod_id in product_ids_seen:
                        continue
                    
                    product_ids_seen.add(prod_id)
                    
                    # Extract description (can be list or string)
                    description = product.get('description', '')
                    if isinstance(description, list):
                        description = ' '.join(str(d) for d in description if d)
                    
                    # Extract categories (can be nested lists)
                    categories = product.get('categories', product.get('category', []))
                    if isinstance(categories, list):
                        # Flatten if nested
                        flat_cats = []
                        for cat in categories:
                            if isinstance(cat, list):
                                flat_cats.extend([str(c) for c in cat if c])
                            else:
                                flat_cats.append(str(cat) if cat else '')
                        categories = flat_cats
                    
                    products.append({
                        'product_id': prod_id,
                        'title': product.get('title', ''),
                        'description': description,
                        'categories': categories,
                        'brand': product.get('brand', ''),
                        'price': product.get('price', 0.0),
                        'features': product.get('features', [])
                    })
                    
                except Exception as e:
                    continue
        
        df = pd.DataFrame(products)
        print(f"✓ Loaded {len(df):,} products")
        print(f"  - Unique product IDs: {df['product_id'].nunique():,}")
        
        return df
