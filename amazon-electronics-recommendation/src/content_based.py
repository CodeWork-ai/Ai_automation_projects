"""
FIXED Content-Based Filtering
✅ Correct similarity calculation
✅ No NaN scores
✅ Proper recommendations
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

class ContentBasedRecommender:
    """
    Content-Based Filtering with TF-IDF and LSA
    """
    
    def __init__(self, max_features=15000, use_lsa=True, lsa_components=100):
        self.max_features = max_features
        self.use_lsa = use_lsa
        self.lsa_components = lsa_components
        self.vectorizer = None
        self.lsa_model = None
        self.tfidf_matrix = None
        self.product_ids = None
        self.products_df = None  # Store for category filtering
    
    def fit(self, products):
        """Fit with advanced NLP techniques"""
        print("Building enhanced TF-IDF matrix...")
        
        valid_products = products[products['combined_features'].str.len() > 0].copy()
        
        if len(valid_products) == 0:
            raise ValueError("No valid products")
        
        print(f"Using {len(valid_products):,} products")
        
        self.product_ids = valid_products['product_id'].values
        self.products_df = valid_products  # Store for later use
        
        # Enhanced vectorizer configuration
        vectorizer_configs = [
            {
                'max_features': self.max_features,
                'ngram_range': (1, 3),
                'min_df': 2,
                'max_df': 0.85,
                'token_pattern': r'(?u)\b[a-zA-Z]\w+\b',
                'lowercase': True,
                'stop_words': 'english',
                'sublinear_tf': True,
                'norm': 'l2',
                'use_idf': True,
                'smooth_idf': True
            },
            # Fallback
            {
                'max_features': self.max_features,
                'ngram_range': (1, 2),
                'min_df': 1,
                'max_df': 1.0,
                'token_pattern': r'(?u)\b\w+\b',
                'lowercase': True
            }
        ]
        
        for i, config in enumerate(vectorizer_configs):
            try:
                self.vectorizer = TfidfVectorizer(**config)
                self.tfidf_matrix = self.vectorizer.fit_transform(
                    valid_products['combined_features']
                )
                
                vocab_size = len(self.vectorizer.vocabulary_)
                print(f"✓ TF-IDF Matrix: {self.tfidf_matrix.shape}")
                print(f"✓ Vocabulary: {vocab_size:,} words")
                
                # Apply LSA for semantic understanding
                if self.use_lsa and self.tfidf_matrix.shape[1] > self.lsa_components:
                    print(f"Applying LSA ({self.lsa_components} components)...")
                    self.lsa_model = TruncatedSVD(
                        n_components=self.lsa_components,
                        random_state=42
                    )
                    self.tfidf_matrix = self.lsa_model.fit_transform(self.tfidf_matrix)
                    variance_explained = self.lsa_model.explained_variance_ratio_.sum()
                    print(f"✓ LSA applied: {variance_explained:.2%} variance explained")
                
                break
                
            except ValueError as e:
                if i < len(vectorizer_configs) - 1:
                    print(f"⚠ Config {i+1} failed, trying fallback...")
                else:
                    raise ValueError(f"All configurations failed: {e}")
    
    def get_similar_products(self, product_id, top_n=10, diversity_boost=0.3):
        """
        Get similar products - FIXED VERSION
        
        Parameters:
        -----------
        product_id : str
            Product ID to find similar products for
        top_n : int
            Number of recommendations
        diversity_boost : float
            Not used anymore - removed buggy diversity filter
        
        Returns:
        --------
        list of tuples: [(product_id, similarity_score), ...]
        """
        # Check if product exists
        if self.product_ids is None:
            print("❌ Model not fitted yet!")
            return []
        
        if product_id not in self.product_ids:
            print(f"❌ Product '{product_id}' not found in model!")
            return []
        
        # Get product index
        idx = np.where(self.product_ids == product_id)[0][0]
        
        # Get product vector
        product_vector = self.tfidf_matrix[idx:idx+1]  # Keep as 2D array
        
        # Calculate cosine similarity with ALL products
        similarities = cosine_similarity(product_vector, self.tfidf_matrix).flatten()
        
        # Get indices sorted by similarity (descending)
        # [1:] to exclude the product itself (which has similarity = 1.0)
        similar_indices = similarities.argsort()[::-1][1:top_n+1]
        
        # Build recommendation list with actual similarity scores
        recommendations = []
        for i in similar_indices:
            product_id_rec = self.product_ids[i]
            similarity_score = float(similarities[i])  # Convert to Python float
            
            # Only add if similarity is meaningful (> 0)
            if similarity_score > 0:
                recommendations.append((product_id_rec, similarity_score))
        
        return recommendations[:top_n]
    
    def get_recommendations(self, user_liked_products, n=10):
        """
        Get recommendations based on multiple products user liked
        
        Parameters:
        -----------
        user_liked_products : list of str
            List of product IDs user liked
        n : int
            Number of recommendations
        
        Returns:
        --------
        list of tuples: [(product_id, score), ...]
        """
        if not user_liked_products:
            return []
        
        # Get indices for all liked products
        valid_indices = []
        for pid in user_liked_products:
            if pid in self.product_ids:
                idx = np.where(self.product_ids == pid)[0][0]
                valid_indices.append(idx)
        
        if not valid_indices:
            return []
        
        # Average the vectors of liked products
        avg_vector = self.tfidf_matrix[valid_indices].mean(axis=0)
        
        # Calculate similarity
        similarities = cosine_similarity(avg_vector.reshape(1, -1), self.tfidf_matrix).flatten()
        
        # Get top N (excluding the liked products themselves)
        similar_indices = similarities.argsort()[::-1]
        
        recommendations = []
        for i in similar_indices:
            if self.product_ids[i] not in user_liked_products:
                recommendations.append((
                    self.product_ids[i],
                    float(similarities[i])
                ))
                
                if len(recommendations) >= n:
                    break
        
        return recommendations
