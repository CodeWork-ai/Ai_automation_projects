"""
Hybrid Recommendation Model - FOR SIMILAR USERS
Combines Collaborative Filtering and Content-Based Filtering
Optimized for showing overlap when users have similar tastes
"""

import numpy as np
from collections import defaultdict

class HybridRecommender:
    def __init__(self, cf_model, content_model, cf_weight=0.7, content_weight=0.3):
        """
        Initialize hybrid recommender
        CHANGED: 70% CF + 30% Content to prioritize similar user patterns
        """
        self.cf_model = cf_model
        self.content_model = content_model
        self.cf_weight = cf_weight
        self.content_weight = content_weight
        self.interactions = None
        self.products = None
        
    def prepare_data(self, interactions, products):
        """Store data for recommendations"""
        self.interactions = interactions
        self.products = products
    def train(self):
   
        """
        Train the hybrid model (hybrid doesn't need training, 
        but we call it for consistency in main.py)
        """
        print("🔄 Preparing Hybrid Model...")
        if self.interactions is None or self.products is None:
            raise ValueError("Call prepare_data() first!")
        print("✅ Hybrid Model ready (both CF and Content-Based are already trained)")
    def recommend(self, user_id, n=10):
        """Generate hybrid recommendations - prioritizes CF for similar users"""
        user_ratings = self.interactions[self.interactions['user_id'] == user_id]
        
        if user_ratings.empty:
            return []
        
        # Get CF recommendations (primary signal)
        try:
            cf_recs = self.cf_model.get_top_n_recommendations(user_id, n=n*2)
        except:
            cf_recs = []
        
        all_candidates = {}
        cf_scores = defaultdict(float)
        content_scores = defaultdict(float)
        
        # Add CF recommendations with HIGH weight (70%)
        for rank, (item_id, cf_score) in enumerate(cf_recs, 1):
            all_candidates[item_id] = cf_score
            cf_scores[item_id] = cf_score
        
        # Add content-based recommendations (secondary signal, 30%)
        seed_products = user_ratings[user_ratings['rating'] >= 4.0]['product_id'].tolist()
        
        if len(seed_products) == 0:
            seed_products = user_ratings['product_id'].tolist()
        
        for user_product in seed_products:
            try:
                similar_items = self.content_model.get_similar_products(user_product, top_n=20)
                for rank, (similar_id, similarity) in enumerate(similar_items, 1):
                    if similar_id not in all_candidates:
                        all_candidates[similar_id] = 0.0
                    content_scores[similar_id] += similarity / len(seed_products)
            except:
                continue
        
        # Calculate final scores: 70% CF + 30% Content
        final_scores = {}
        for item_id in all_candidates:
            cf_score = cf_scores.get(item_id, 0.0)
            content_score = content_scores.get(item_id, 0.0)
            
            # 70% CF weight + 30% Content weight
            final_scores[item_id] = (
                self.cf_weight * cf_score +
                self.content_weight * content_score
            )
        
        # Filter out already rated products
        if self.interactions is not None:
            user_rated = set(user_ratings['product_id'].tolist())
            final_scores = {k: v for k, v in final_scores.items() if k not in user_rated}
        
        # Sort by score
        sorted_items = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        
        # No aggressive diversity filter - let top CF recommendations appear
        recommendations = []
        category_counts = defaultdict(int)
        
        for product_id, score in sorted_items:
            if len(recommendations) >= n:
                break
            
            product_info = self.products[self.products['product_id'] == product_id]
            if product_info.empty:
                continue
            
            # Allow more products per category (10 instead of 5)
            category = product_info.iloc[0].get('main_category', 'Unknown')
            if category_counts[category] < 10:
                recommendations.append((product_id, score))
                category_counts[category] += 1
        
        return recommendations
    def evaluate(self):
        """Evaluate hybrid model"""
        print("📊 Evaluating Hybrid Model...")
        
        # Get unique users from interactions
        users = self.interactions['user_id'].unique()
        
        # Calculate metrics
        precision_sum = 0
        recall_sum = 0
        auc_sum = 0
        count = 0
        
        for user in users[:100]:  # Sample 100 users for evaluation
            try:
                recommendations = self.recommend(user, n=5)
                
                if len(recommendations) > 0:
                    # Get user's rated products
                    user_ratings = self.interactions[self.interactions['user_id'] == user]
                    highly_rated = set(user_ratings[user_ratings['rating'] >= 4.0]['product_id'].tolist())
                    
                    rec_items = set([r[0] for r in recommendations])
                    
                    if len(highly_rated) > 0:
                        precision = len(rec_items & highly_rated) / len(rec_items) if len(rec_items) > 0 else 0
                        recall = len(rec_items & highly_rated) / len(highly_rated)
                        auc = min(1.0, precision + recall) / 2
                        
                        precision_sum += precision
                        recall_sum += recall
                        auc_sum += auc
                        count += 1
            except:
                continue
        
        if count == 0:
            return 0.0, 0.0, 0.0
        
        precision = precision_sum / count
        recall = recall_sum / count
        auc = auc_sum / count
        
        print(f" Precision@5: {precision:.4f}")
        print(f" Recall@5: {recall:.4f}")
        print(f" AUC: {auc:.4f}")
        
        return precision, recall, auc