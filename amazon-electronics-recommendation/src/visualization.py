"""
Visualization Module for Hybrid Recommendation System
Generates comprehensive charts using matplotlib and seaborn
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from collections import defaultdict
import os

class RecommendationVisualizer:
    """Visualize recommendation system metrics and insights using matplotlib & seaborn"""
    
    def __init__(self, results_dir='results'):
        """Initialize visualizer with style settings"""
        sns.set_style("whitegrid")
        sns.set_palette("husl")
        plt.rcParams['figure.figsize'] = (14, 8)
        plt.rcParams['font.size'] = 10
        
        # Create results directory if it doesn't exist
        self.results_dir = results_dir
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
            print(f"✅ Created {self.results_dir}/ directory")
    
    def plot_model_performance(self, results):
        """Compare RMSE, MAE, Precision, Recall, AUC across models"""
        print("📊 Generating model performance chart...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Model Performance Comparison', fontsize=14, fontweight='bold', y=1.02)
        
        # Get data from results
        cf_data = results.get('Collaborative_Filtering', {})
        hybrid_data = results.get('Hybrid', {})
        
        # Error Metrics (RMSE, MAE)
        models = ['Collaborative\nFiltering', 'Hybrid\nModel']
        rmse_vals = [cf_data.get('RMSE', 0), hybrid_data.get('Precision@5', 0)]
        mae_vals = [cf_data.get('MAE', 0), hybrid_data.get('Recall@5', 0)]
        
        x = np.arange(len(models))
        width = 0.35
        
        bars1 = axes[0].bar(x - width/2, rmse_vals, width, label='RMSE', color='#FF6B6B', edgecolor='black', linewidth=1.5)
        bars2 = axes[0].bar(x + width/2, mae_vals, width, label='MAE', color='#4ECDC4', edgecolor='black', linewidth=1.5)
        
        axes[0].set_ylabel('Error Score', fontsize=11, fontweight='bold')
        axes[0].set_title('Error Metrics', fontsize=12, fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(models, fontsize=10)
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                axes[0].text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.4f}', ha='center', va='bottom', fontsize=9)
        
        # Ranking Metrics (Precision, Recall, AUC)
        precision = hybrid_data.get('Precision@5', 0)
        recall = hybrid_data.get('Recall@5', 0)
        auc = hybrid_data.get('AUC', 0)
        
        metrics = ['Precision@5', 'Recall@5', 'AUC']
        values = [precision, recall, auc]
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        bars = axes[1].barh(metrics, values, color=colors, edgecolor='black', linewidth=1.5)
        axes[1].set_xlabel('Score', fontsize=11, fontweight='bold')
        axes[1].set_title('Ranking Quality Metrics', fontsize=12, fontweight='bold')
        axes[1].set_xlim(0, 1)
        axes[1].grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (bar, v) in enumerate(zip(bars, values)):
            axes[1].text(v + 0.02, i, f'{v:.4f}', va='center', fontweight='bold', fontsize=10)
        
        plt.tight_layout()
        save_path = os.path.join(self.results_dir, 'model_performance.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {save_path}")
    
    def plot_rating_distribution(self, interactions):
        """Visualize rating distribution"""
        print("📊 Generating rating distribution chart...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Rating Distribution Analysis', fontsize=14, fontweight='bold', y=1.02)
        
        # Rating histogram
        axes[0].hist(interactions['rating'], bins=20, color='#3498db', edgecolor='black', alpha=0.8, linewidth=1.5)
        axes[0].set_xlabel('Rating', fontsize=11, fontweight='bold')
        axes[0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
        axes[0].set_title('Distribution of User Ratings', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='y')
        axes[0].axvline(interactions['rating'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {interactions["rating"].mean():.2f}')
        axes[0].legend()
        
        # Rating count by score
        rating_counts = interactions['rating'].value_counts().sort_index()
        axes[1].plot(rating_counts.index, rating_counts.values, marker='o', linewidth=2.5, markersize=10, color='#27ae60', markerfacecolor='#2ecc71', markeredgewidth=2)
        axes[1].fill_between(rating_counts.index, rating_counts.values, alpha=0.3, color='#2ecc71')
        axes[1].set_xlabel('Rating Score', fontsize=11, fontweight='bold')
        axes[1].set_ylabel('Count', fontsize=11, fontweight='bold')
        axes[1].set_title('Ratings by Score', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xticks(sorted(rating_counts.index))
        
        plt.tight_layout()
        save_path = os.path.join(self.results_dir, 'rating_distribution.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {save_path}")
    
    def plot_user_item_matrix_heatmap(self, interactions, sample_users=30, sample_items=30):
        """Visualize sparse user-item interaction matrix"""
        print("📊 Generating user-item heatmap...")
        
        try:
            # Create a sample pivot table
            sample_data = interactions.sample(n=min(len(interactions), 3000))
            pivot = sample_data.pivot_table(index='user_id', columns='product_id', values='rating', fill_value=0)
            
            # Sample subset for visualization
            pivot = pivot.iloc[:min(sample_users, len(pivot)), :min(sample_items, len(pivot.columns))]
            
            fig, ax = plt.subplots(figsize=(14, 8))
            sns.heatmap(pivot, cmap='YlOrRd', cbar_kws={'label': 'Rating'}, ax=ax, 
                       xticklabels=False, yticklabels=False, linewidths=0.5)
            ax.set_title('User-Item Interaction Matrix (Sparse)', fontsize=14, fontweight='bold', pad=20)
            ax.set_xlabel('Products', fontsize=11, fontweight='bold')
            ax.set_ylabel('Users', fontsize=11, fontweight='bold')
            
            plt.tight_layout()
            save_path = os.path.join(self.results_dir, 'user_item_heatmap.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ Saved: {save_path}")
        except Exception as e:
            print(f"⚠️ Error generating heatmap: {e}")
    
    def plot_sparsity_analysis(self, interactions):
        """Analyze and visualize data sparsity"""
        print("📊 Generating sparsity analysis chart...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Data Sparsity Analysis', fontsize=14, fontweight='bold', y=1.02)
        
        # Calculate sparsity
        total_possible = interactions['user_id'].nunique() * interactions['product_id'].nunique()
        actual_interactions = len(interactions)
        sparsity = (1 - actual_interactions / total_possible) * 100
        
        # Sparsity pie chart
        sizes = [actual_interactions, total_possible - actual_interactions]
        colors = ['#2ecc71', '#e74c3c']
        explode = (0.1, 0)
        wedges, texts, autotexts = axes[0].pie(sizes, labels=['Interactions', 'Missing'], autopct='%1.2f%%', 
                                                colors=colors, startangle=90, explode=explode,
                                                textprops={'fontsize': 10, 'fontweight': 'bold'})
        axes[0].set_title(f'Data Sparsity: {sparsity:.2f}%', fontsize=12, fontweight='bold')
        
        # Interactions per user histogram
        user_interactions = interactions['user_id'].value_counts()
        axes[1].hist(user_interactions, bins=50, color='#9b59b6', edgecolor='black', alpha=0.8, linewidth=1.5)
        axes[1].set_xlabel('Interactions per User', fontsize=11, fontweight='bold')
        axes[1].set_ylabel('Number of Users', fontsize=11, fontweight='bold')
        axes[1].set_title('User Activity Distribution', fontsize=12, fontweight='bold')
        
        mean_val = user_interactions.mean()
        median_val = user_interactions.median()
        axes[1].axvline(mean_val, color='red', linestyle='--', linewidth=2.5, label=f'Mean: {mean_val:.1f}')
        axes[1].axvline(median_val, color='orange', linestyle='-.', linewidth=2.5, label=f'Median: {median_val:.1f}')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        save_path = os.path.join(self.results_dir, 'sparsity_analysis.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {save_path}")
    
    def plot_recommendation_quality(self, recommendations, interactions):
        """Visualize recommendation quality metrics"""
        print("📊 Generating recommendation quality chart...")
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle('Recommendation Quality Metrics', fontsize=14, fontweight='bold', y=0.995)
            
            # Diversity analysis
            diversity_scores = []
            for recs in recommendations.values():
                if len(recs) > 0:
                    unique_items = len(set([r[0] for r in recs]))
                    diversity_scores.append(unique_items / len(recs))
            
            if diversity_scores:
                axes[0, 0].hist(diversity_scores, bins=30, color='#e74c3c', edgecolor='black', alpha=0.8, linewidth=1.5)
                axes[0, 0].set_xlabel('Diversity Score', fontsize=10, fontweight='bold')
                axes[0, 0].set_ylabel('Frequency', fontsize=10, fontweight='bold')
                axes[0, 0].set_title('Recommendation Diversity Distribution', fontsize=11, fontweight='bold')
                axes[0, 0].grid(True, alpha=0.3, axis='y')
                axes[0, 0].axvline(np.mean(diversity_scores), color='blue', linestyle='--', linewidth=2, label=f'Mean: {np.mean(diversity_scores):.2f}')
                axes[0, 0].legend()
            
            # Coverage analysis
            all_recommended = set()
            for recs in recommendations.values():
                all_recommended.update([r[0] for r in recs])
            
            total_products = interactions['product_id'].nunique()
            coverage = len(all_recommended) / total_products * 100
            
            bars = axes[0, 1].bar(['Covered', 'Not Covered'], [coverage, 100-coverage], 
                                 color=['#1abc9c', '#95a5a6'], edgecolor='black', linewidth=1.5)
            axes[0, 1].set_ylabel('Percentage (%)', fontsize=10, fontweight='bold')
            axes[0, 1].set_title(f'Catalog Coverage: {coverage:.1f}%', fontsize=11, fontweight='bold')
            axes[0, 1].set_ylim(0, 100)
            axes[0, 1].grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                              f'{height:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
            
            # Recommendation list length distribution
            rec_lengths = [len(recs) for recs in recommendations.values()]
            axes[1, 0].hist(rec_lengths, bins=20, color='#3498db', edgecolor='black', alpha=0.8, linewidth=1.5)
            axes[1, 0].set_xlabel('Number of Recommendations', fontsize=10, fontweight='bold')
            axes[1, 0].set_ylabel('Frequency', fontsize=10, fontweight='bold')
            axes[1, 0].set_title('Recommendation List Length Distribution', fontsize=11, fontweight='bold')
            axes[1, 0].grid(True, alpha=0.3, axis='y')
            axes[1, 0].axvline(np.mean(rec_lengths), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(rec_lengths):.1f}')
            axes[1, 0].legend()
            
            # Score distribution
            all_scores = []
            for recs in recommendations.values():
                all_scores.extend([r[2] if len(r) > 2 else 0 for r in recs])
            
            if all_scores:
                axes[1, 1].hist(all_scores, bins=40, color='#f39c12', edgecolor='black', alpha=0.8, linewidth=1.5)
                axes[1, 1].set_xlabel('Recommendation Score', fontsize=10, fontweight='bold')
                axes[1, 1].set_ylabel('Frequency', fontsize=10, fontweight='bold')
                axes[1, 1].set_title('Recommendation Score Distribution', fontsize=11, fontweight='bold')
                axes[1, 1].grid(True, alpha=0.3, axis='y')
                axes[1, 1].axvline(np.mean(all_scores), color='green', linestyle='--', linewidth=2, label=f'Mean: {np.mean(all_scores):.2f}')
                axes[1, 1].legend()
            
            plt.tight_layout()
            save_path = os.path.join(self.results_dir, 'recommendation_quality.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ Saved: {save_path}")
        except Exception as e:
            print(f"⚠️ Error generating recommendation quality chart: {e}")
    
    def plot_top_rated_products(self, interactions, products_df, top_n=15):
        """Visualize top-rated products"""
        print("📊 Generating top-rated products chart...")
        
        try:
            # Get average rating per product
            avg_ratings = interactions.groupby('product_id')['rating'].agg(['mean', 'count']).reset_index()
            avg_ratings = avg_ratings[avg_ratings['count'] >= 5]  # Filter by minimum reviews
            avg_ratings = avg_ratings.nlargest(top_n, 'mean')
            
            fig, ax = plt.subplots(figsize=(12, 6))
            bars = ax.barh(range(len(avg_ratings)), avg_ratings['mean'].values, color='#16a085', edgecolor='black', linewidth=1.5)
            ax.set_yticks(range(len(avg_ratings)))
            ax.set_yticklabels([f"Product {pid}" for pid in avg_ratings['product_id'].values], fontsize=10)
            ax.set_xlabel('Average Rating', fontsize=11, fontweight='bold')
            ax.set_title(f'Top {top_n} Rated Products (min 5 reviews)', fontsize=12, fontweight='bold')
            ax.set_xlim(0, 5.5)
            ax.grid(True, alpha=0.3, axis='x')
            
            # Add value labels
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax.text(width + 0.1, bar.get_y() + bar.get_height()/2.,
                       f'{width:.2f} ({int(avg_ratings.iloc[i]["count"])} reviews)',
                       ha='left', va='center', fontweight='bold', fontsize=9)
            
            plt.tight_layout()
            save_path = os.path.join(self.results_dir, 'top_rated_products.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ Saved: {save_path}")
        except Exception as e:
            print(f"⚠️ Error generating top-rated products chart: {e}")

