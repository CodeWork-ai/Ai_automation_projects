"""
Script to find two users with similar tastes in your dataset
"""

import pandas as pd
from src.data_loader import DataLoader
from src.preprocessing import DataPreprocessor
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def find_similar_users():
    print("Loading data...")
    
    # Load data
    loader = DataLoader(
        ratings_path='data/Electronics.jsonl',
        metadata_path='data/meta_Electronics.jsonl'
    )
    
    interactions = loader.load_ratings(sample_size=300000)
    products = loader.load_metadata()
    
    # Preprocess
    preprocessor = DataPreprocessor()
    products = preprocessor.prepare_metadata(products)
    
    common_products = set(products['product_id']) & set(interactions['product_id'])
    interactions = interactions[interactions['product_id'].isin(common_products)]
    
    interactions = preprocessor.filter_sparse_data(
        interactions,
        min_user_interactions=2,
        min_product_interactions=2
    )
    
    print(f"Total users: {interactions['user_id'].nunique()}")
    print(f"Total products: {interactions['product_id'].nunique()}")
    
    # Create user-item matrix for similarity calculation
    print("\nCreating user-item matrix...")
    user_item_matrix = interactions.pivot_table(
        index='user_id',
        columns='product_id',
        values='rating',
        fill_value=0
    )
    
    print(f"Matrix shape: {user_item_matrix.shape}")
    
    # Calculate user similarity (sample first 100 users for speed)
    print("\nCalculating user similarities (sampling 100 users)...")
    sample_users = user_item_matrix.head(100)
    similarity_matrix = cosine_similarity(sample_users)
    
    # Find most similar user pairs
    print("\nFinding similar user pairs...")
    similar_pairs = []
    
    for i in range(len(similarity_matrix)):
        for j in range(i+1, len(similarity_matrix)):
            if similarity_matrix[i][j] > 0.3:  # Similarity threshold
                similar_pairs.append({
                    'user_a': sample_users.index[i],
                    'user_b': sample_users.index[j],
                    'similarity': similarity_matrix[i][j]
                })
    
    # Sort by similarity
    similar_pairs = sorted(similar_pairs, key=lambda x: x['similarity'], reverse=True)
    
    print("\n" + "="*80)
    print("TOP 5 MOST SIMILAR USER PAIRS")
    print("="*80)
    
    for idx, pair in enumerate(similar_pairs[:5], 1):
        user_a = pair['user_a']
        user_b = pair['user_b']
        similarity = pair['similarity']
        
        print(f"\n{idx}. User A: {user_a} | User B: {user_b} | Similarity: {similarity:.2%}")
        
        # Show what they both rated
        user_a_ratings = interactions[interactions['user_id'] == user_a]
        user_b_ratings = interactions[interactions['user_id'] == user_b]
        
        print(f"   User A rated {len(user_a_ratings)} products")
        print(f"   User B rated {len(user_b_ratings)} products")
        
        # Find common products
        common_products = set(user_a_ratings['product_id']) & set(user_b_ratings['product_id'])
        print(f"   Common products rated: {len(common_products)}")
    
    if similar_pairs:
        print("\n" + "="*80)
        print("RECOMMENDATION FOR TESTING:")
        print("="*80)
        best_pair = similar_pairs[0]
        print(f"User A ID: {best_pair['user_a']}")
        print(f"User B ID: {best_pair['user_b']}")
        print(f"Similarity: {best_pair['similarity']:.2%}")
        print(f"\nExpected result: Should see 30-60% overlap in recommendations")
        print("(Because they have similar tastes)")
    
    return similar_pairs

if __name__ == "__main__":
    find_similar_users()
