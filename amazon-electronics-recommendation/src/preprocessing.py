import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import re

class DataPreprocessor:
    """Preprocess and clean Amazon Electronics data"""
    
    def __init__(self):
        self.user_encoder = LabelEncoder()
        self.product_encoder = LabelEncoder()
    
    def filter_sparse_data(self, interactions, min_user_interactions=3, min_product_interactions=3):
        """Filter out users and products with too few interactions"""
        print("Filtering sparse data...")
        
        user_counts = interactions['user_id'].value_counts()
        product_counts = interactions['product_id'].value_counts()
        
        valid_users = user_counts[user_counts >= min_user_interactions].index
        valid_products = product_counts[product_counts >= min_product_interactions].index
        
        filtered = interactions[
            (interactions['user_id'].isin(valid_users)) & 
            (interactions['product_id'].isin(valid_products))
        ]
        
        print(f"Filtered from {len(interactions)} to {len(filtered)} interactions")
        return filtered
    
    def encode_ids(self, interactions):
        """Encode user and product IDs to integers"""
        print("Encoding user and product IDs...")
        
        interactions['user_idx'] = self.user_encoder.fit_transform(interactions['user_id'])
        interactions['product_idx'] = self.product_encoder.fit_transform(interactions['product_id'])
        
        return interactions
    
    def clean_text(self, text):
        """Clean text by removing special characters and extra spaces"""
        if not isinstance(text, str):
            return ''
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        # Remove special characters but keep alphanumeric and spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        # Remove extra spaces
        text = ' '.join(text.split())
        
        return text
    
    def prepare_metadata(self, products):
        """Clean and prepare product metadata - MORE LENIENT VERSION"""
        print("Preparing product metadata...")
        
        # Fill missing values
        products['title'] = products['title'].fillna('').astype(str)
        products['brand'] = products['brand'].fillna('Unknown').astype(str)
        
        # Handle description field
        def process_description(desc):
            if isinstance(desc, list):
                return ' '.join(str(item) for item in desc if item)
            elif isinstance(desc, str):
                return desc
            else:
                return ''
        
        products['description'] = products['description'].apply(process_description)
        
        # Handle categories field
        def process_categories(cats):
            if isinstance(cats, list):
                flat_cats = []
                for cat in cats:
                    if isinstance(cat, list):
                        flat_cats.extend([str(c) for c in cat if c])
                    else:
                        flat_cats.append(str(cat) if cat else '')
                return ' '.join(flat_cats)
            elif isinstance(cats, str):
                return cats
            else:
                return ''
        
        products['categories'] = products['categories'].apply(process_categories)
        
        # Clean text fields
        products['title'] = products['title'].apply(self.clean_text)
        products['description'] = products['description'].apply(self.clean_text)
        products['categories'] = products['categories'].apply(self.clean_text)
        products['brand'] = products['brand'].apply(self.clean_text)
        
        # Create combined features - if title is empty, use product_id
        def create_combined_features(row):
            features = []
            if row['title']:
                features.append(row['title'])
            if row['description']:
                features.append(row['description'])
            if row['categories']:
                features.append(row['categories'])
            if row['brand'] and row['brand'] != 'Unknown':
                features.append(row['brand'])
            
            # If all empty, use product_id as fallback
            if not features:
                return f"product {row['product_id']}"
            
            return ' '.join(features)
        
        products['combined_features'] = products.apply(create_combined_features, axis=1)
        
        # Only remove completely empty products (very lenient)
        products = products[products['combined_features'].str.strip() != '']
        
        print(f"Prepared {len(products)} products with metadata")
        
        return products
