import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

class ModelEvaluator:
    """Comprehensive evaluation for all recommendation models"""
    
    def __init__(self):
        self.results = {}
    
    def evaluate_collaborative_filtering(self, model):
        """Evaluate Implicit collaborative filtering model"""
        print("Evaluating Collaborative Filtering Model...")
        
        rmse, mae, predictions = model.evaluate()
        
        self.results['Collaborative_Filtering'] = {
            'RMSE': rmse,
            'MAE': mae
        }
        
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}")
        
        return self.results['Collaborative_Filtering']
    
    def evaluate_hybrid(self, model):
        """Evaluate hybrid model"""
        print("Evaluating Hybrid Model...")
        
        precision, recall, auc = model.evaluate()
        
        self.results['Hybrid'] = {
            'Precision@5': precision,
            'Recall@5': recall,
            'AUC': auc
        }
        
        print(f"  Precision@5: {precision:.4f}")
        print(f"  Recall@5: {recall:.4f}")
        print(f"  AUC: {auc:.4f}")
        
        return self.results['Hybrid']
    
    def evaluate_content_based(self, recommender, products, sample_size=100):
        """Evaluate content-based filtering"""
        print("Evaluating Content-Based Filtering...")
        
        sample_products = products.sample(n=min(sample_size, len(products)))
        
        similarity_scores = []
        recommendation_counts = []
        
        for product_id in sample_products['product_id']:
            similar = recommender.get_similar_products(product_id, top_n=10)
            
            if similar:
                avg_score = np.mean([score for _, score in similar])
                similarity_scores.append(avg_score)
                recommendation_counts.append(len(similar))
        
        self.results['Content_Based'] = {
            'Avg_Similarity': np.mean(similarity_scores) if similarity_scores else 0.0,
            'Std_Similarity': np.std(similarity_scores) if similarity_scores else 0.0,
            'Avg_Recommendations': np.mean(recommendation_counts) if recommendation_counts else 0.0,
            'Coverage': len(recommendation_counts) / sample_size if sample_size > 0 else 0.0
        }
        
        print(f"  Average Similarity: {self.results['Content_Based']['Avg_Similarity']:.4f}")
        print(f"  Std Similarity: {self.results['Content_Based']['Std_Similarity']:.4f}")
        print(f"  Coverage: {self.results['Content_Based']['Coverage']:.2%}")
        
        return self.results['Content_Based']
    
    def evaluate_diversity(self, recommendations_list, products):
        """Measure recommendation diversity"""
        if len(recommendations_list) < 2:
            return 0.0
        
        categories = products[products['product_id'].isin(recommendations_list)]['categories'].values
        unique_categories = len(set(' '.join(categories).split()))
        
        diversity_score = unique_categories / len(recommendations_list)
        return diversity_score
    
    def evaluate_novelty(self, recommendations_list, interactions):
        """Measure recommendation novelty"""
        product_popularity = interactions['product_id'].value_counts()
        
        novelty_scores = []
        for product_id in recommendations_list:
            if product_id in product_popularity:
                novelty = 1 / (1 + np.log(1 + product_popularity[product_id]))
                novelty_scores.append(novelty)
        
        return np.mean(novelty_scores) if novelty_scores else 0.0
    
    def compare_models(self):
        """Compare all evaluated models"""
        print("\n" + "=" * 80)
        print("MODEL COMPARISON SUMMARY")
        print("=" * 80)
        
        for model_name, metrics in self.results.items():
            print(f"\n{model_name}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")
        
        return None
    
    def generate_report(self, output_path='evaluation_report.txt'):
        """Generate evaluation report"""
        with open(output_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("AMAZON ELECTRONICS RECOMMENDATION SYSTEM - EVALUATION REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            for model_name, metrics in self.results.items():
                f.write(f"\n{model_name}:\n")
                f.write("-" * 40 + "\n")
                for metric, value in metrics.items():
                    f.write(f"  {metric}: {value:.4f}\n")
        
        print(f"\nEvaluation report saved to: {output_path}")
        return output_path
