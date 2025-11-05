"""
Amazon Electronics Recommendation System - Streamlit UI
Clean interface with product titles and IDs
"""

import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from src.data_loader import DataLoader
from src.preprocessing import DataPreprocessor
from src.content_based import ContentBasedRecommender
from src.collaborative import CollaborativeFilteringRecommender
from src.hybrid_model import HybridRecommender

# Page config
st.set_page_config(
    page_title="Amazon Electronics Recommender",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #232F3E 0%, #FF9900 100%);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 30px;
    }
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2.5rem;
    }
    .main-header p {
        color: white;
        margin: 5px 0 0 0;
        font-size: 1.1rem;
    }
    .rec-card {
        background: white;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #FF9900;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .rec-card h4, .rec-card h5 {
        margin-top: 0;
        margin-bottom: 8px;
        color: #232F3E;
    }
    .rec-card p {
        margin: 5px 0;
        color: #666;
        font-size: 0.9rem;
    }
    .stButton>button {
        background: linear-gradient(90deg, #FF9900 0%, #FF6600 100%);
        color: white;
        font-weight: bold;
        border: none;
        padding: 10px 24px;
        font-size: 16px;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #FF6600 0%, #FF9900 100%);
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data(sample_size=300000):
    """Load and preprocess JSONL data"""
    progress = st.progress(0)
    status = st.empty()
    
    try:
        status.text("📂 Loading ratings...")
        progress.progress(20)
        
        loader = DataLoader(
            ratings_path='data/Electronics.jsonl',
            metadata_path='data/meta_Electronics.jsonl'
        )
        
        interactions = loader.load_ratings(sample_size=sample_size)
        progress.progress(50)
        
        products = loader.load_metadata()
        progress.progress(70)
        
        products_full = products.copy()
        
        preprocessor = DataPreprocessor()
        products = preprocessor.prepare_metadata(products)
        products_full = preprocessor.prepare_metadata(products_full)
        
        common_products = set(products['product_id']) & set(interactions['product_id'])
        interactions = interactions[interactions['product_id'].isin(common_products)]
        products = products[products['product_id'].isin(common_products)]
        
        interactions = preprocessor.filter_sparse_data(
            interactions,
            min_user_interactions=2,
            min_product_interactions=2
        )
        
        interactions = preprocessor.encode_ids(interactions)
        
        progress.progress(100)
        status.text("✅ Data loaded successfully!")
        
        return interactions, products, products_full
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None

@st.cache_resource
def load_models(_interactions, _products, _products_full):
    """Train all recommendation models"""
    models = {}
    
    with st.spinner("🔧 Training Content-Based on full catalog..."):
        try:
            content = ContentBasedRecommender(
                max_features=3000,
                use_lsa=True,
                lsa_components=100
            )
            content.fit(_products_full)
            models['content'] = content
           
        except Exception as e:
            st.error(f"Content-based error: {str(e)}")
            models['content'] = None
    
    with st.spinner("🔧 Training Collaborative Filtering (SVD++)..."):
        try:
            cf = CollaborativeFilteringRecommender(
                n_factors=100,
                n_epochs=20,
                lr_all=0.007,
                reg_all=0.02
            )
            cf.prepare_data(_interactions, test_size=0.2)
            cf.train()
            models['cf'] = cf
           
        except Exception as e:
            st.error(f"Collaborative filtering error: {str(e)}")
            models['cf'] = None
    
    with st.spinner("🔧 Creating Hybrid Model..."):
        try:
            if models['cf'] and models['content']:
                hybrid = HybridRecommender(
                    models['cf'],
                    models['content'],
                    cf_weight=0.5,
                    content_weight=0.5
                )
                hybrid.prepare_data(_interactions, _products_full)
                models['hybrid'] = hybrid
                
            else:
                models['hybrid'] = None
        except Exception as e:
            st.error(f"Hybrid error: {str(e)}")
            models['hybrid'] = None
    
    return models

def main():
    # Header
    st.markdown("""
        <div class="main-header">
            <h1>🛒 Amazon Electronics Recommender</h1>
            <p>AI-Powered Product Recommendations</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("⚙️ Settings")
    
    st.sidebar.markdown("### 📊 Algorithm")
    method = st.sidebar.selectbox(
        "Recommendation Model",
        ["Hybrid (Best)", "Collaborative Filtering", "Content-Based"]
    )
    
    st.sidebar.markdown("### 📈 Results")
    n_recommendations = st.sidebar.slider(
        "Number of recommendations", 
        min_value=5, 
        max_value=20, 
        value=10
    )
    
    show_stats = st.sidebar.checkbox("📊 Show Dataset Stats", value=True)
    
    # Load data
    interactions, products, products_full = load_data(sample_size=300000)
    
    if interactions is None:
        st.error("Failed to load data. Please check data files.")
        return
    
    models = load_models(interactions, products, products_full)
    
    if show_stats:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📊 Dataset Statistics")
        st.sidebar.metric("Total Products", f"{len(products_full):,}")
        st.sidebar.metric("Users", f"{interactions['user_id'].nunique():,}")
        st.sidebar.metric("Ratings", f"{len(interactions):,}")
    
    # Main content
    st.markdown("## 🎯 Get Product Recommendations")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        user_ids = sorted(interactions['user_id'].unique())
        selected_user = st.selectbox(
            "Choose a user",
            user_ids,
            key="user_select"
        )
    
    with col2:
        user_rating_count = len(interactions[interactions['user_id'] == selected_user])
        st.metric("User's Ratings", user_rating_count)
    
    if st.button("🚀 Get Recommendations", key="get_recs_btn", use_container_width=True):
        with st.spinner(f"🔄 Generating recommendations..."):
            try:
                recs = []
                
                if method == "Hybrid (Best)" and models['hybrid']:
                    recs = models['hybrid'].recommend(selected_user, n_recommendations)
                    user_ratings = interactions[interactions['user_id'] == selected_user].sort_values('rating', ascending=False)
                
                    #if not user_ratings.empty:
                       #with st.expander("🔍 Debug: User's Rating History", expanded=False):
                       #st.write(f"**Total rated products:** {len(user_ratings)}")
                       #for idx, row in user_ratings.iterrows():
                            #prod = products_full[products_full['product_id'] == row['product_id']]
                            #if not prod.empty:
                                #st.write(f"⭐ {row['rating']:.1f} - {prod.iloc[0]['title']}")
                
                    recs = models['hybrid'].recommend(selected_user, n_recommendations)
                elif method == "Collaborative Filtering" and models['cf']:
                    recs = models['cf'].get_top_n_recommendations(selected_user, n=n_recommendations)
                    
                elif method == "Content-Based" and models['content']:
                    user_ratings = interactions[interactions['user_id'] == selected_user].sort_values('rating', ascending=False)
                    
                    if not user_ratings.empty:
                        top_product = user_ratings.iloc[0]['product_id']
                        prod_info = products_full[products_full['product_id'] == top_product]
                        if not prod_info.empty:
                           st.info(f"🔍 **Content-Based using this product as seed:**")
                           st.write(f"⭐ {user_ratings.iloc[0]['rating']:.1f} - {prod_info.iloc[0]['title']}")
                    
                        recs = models['content'].get_similar_products(top_product, top_n=n_recommendations)
                    else:
                        st.warning("⚠️ No rating history found")
                        recs = []
                else:
                    st.error(f"❌ {method} model is not available")
                    recs = []
                
                if recs:
                    st.success(f"✅ Generated {len(recs)} recommendations!")
                    st.markdown("---")
                    st.markdown(f"### 🎁 Top {len(recs)} Recommendations")
                    
                    for idx, (product_id, score) in enumerate(recs, 1):
                        product_info = products_full[products_full['product_id'] == product_id]
                        
                        if not product_info.empty:
                            row = product_info.iloc[0]
                            title = row.get('title', 'Unknown Product')
                            
                            st.markdown(f"""
                            <div class="rec-card">
                                <h4>#{idx} {title}</h4>
                                <p><strong>Product ID:</strong> {product_id}</p>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.warning("⚠️ No recommendations generated.")
                    
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    # Compare Users Section
    st.markdown("---")
    st.markdown("## 🔄 Compare Two Users")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown("### 👤 User A")
        user_a = st.selectbox(
            "Select User A",
            user_ids,
            key="user_a_select",
            index=0
        )
        user_a_ratings = interactions[interactions['user_id'] == user_a]
        st.metric("User A's Ratings", len(user_a_ratings))
        
        if not user_a_ratings.empty:
            top_rated_a = user_a_ratings.nlargest(3, 'rating')
            st.markdown("**Top 3 Rated Products:**")
            for _, row in top_rated_a.iterrows():
                prod = products_full[products_full['product_id'] == row['product_id']]
                if not prod.empty:
                    st.write(f"⭐ {row['rating']:.1f} - {prod.iloc[0]['title'][:50]}...")
    
    with col_b:
        st.markdown("### 👤 User B")
        user_b = st.selectbox(
            "Select User B",
            user_ids,
            key="user_b_select",
            index=min(10, len(user_ids)-1)
        )
        user_b_ratings = interactions[interactions['user_id'] == user_b]
        st.metric("User B's Ratings", len(user_b_ratings))
        
        if not user_b_ratings.empty:
            top_rated_b = user_b_ratings.nlargest(3, 'rating')
            st.markdown("**Top 3 Rated Products:**")
            for _, row in top_rated_b.iterrows():
                prod = products_full[products_full['product_id'] == row['product_id']]
                if not prod.empty:
                    st.write(f"⭐ {row['rating']:.1f} - {prod.iloc[0]['title'][:50]}...")
    
    if st.button("🔍 Compare Recommendations", key="compare_btn", use_container_width=True):
        with st.spinner("Generating recommendations for both users..."):
            try:
                if models['hybrid']:
                    recs_a = models['hybrid'].recommend(user_a, 20)
                    recs_b = models['hybrid'].recommend(user_b, 20)

                    st.success("✅ Generated hybrid recommendations for both users!")
                    st.markdown("---")
                    
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        st.markdown("### 🎁 Recommendations for User A")
                        for idx, (product_id, score) in enumerate(recs_a, 1):
                            product_info = products_full[products_full['product_id'] == product_id]
                            if not product_info.empty:
                                row = product_info.iloc[0]
                                st.markdown(f"""
                                <div class="rec-card">
                                    <h5>#{idx} {row.get('title', 'Unknown')}</h5>
                                    <p><strong>Product ID:</strong> {product_id}</p>
                                </div>
                                """, unsafe_allow_html=True)
                    
                    with col_b:
                        st.markdown("### 🎁 Recommendations for User B")
                        for idx, (product_id, score) in enumerate(recs_b, 1):
                            product_info = products_full[products_full['product_id'] == product_id]
                            if not product_info.empty:
                                row = product_info.iloc[0]
                                st.markdown(f"""
                                <div class="rec-card">
                                    <h5>#{idx} {row.get('title', 'Unknown')}</h5>
                                    <p><strong>Product ID:</strong> {product_id}</p>
                                </div>
                                """, unsafe_allow_html=True)
                    
                    st.markdown("---")
                    st.markdown("### 📊 Comparison Analysis")
                    
                    recs_a_ids = set([p[0] for p in recs_a])
                    recs_b_ids = set([p[0] for p in recs_b])
                    overlap = recs_a_ids & recs_b_ids
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("User A Unique", len(recs_a_ids - recs_b_ids))
                    col2.metric("Overlap", len(overlap))
                    col3.metric("User B Unique", len(recs_b_ids - recs_a_ids))
                    
                    if len(overlap) == 0:
                        st.success("✅ **Highly personalized!** No overlap - completely different recommendations!")
                    elif len(overlap) <= 2:
                        st.info("✅ **Good personalization!** Minimal overlap - tailored recommendations.")
                    else:
                        st.warning("⚠️ Some overlap detected.")
                    
                    if len(overlap) > 0:
                        st.markdown("**Overlapping Recommendations:**")
                        for product_id in overlap:
                            prod = products_full[products_full['product_id'] == product_id]
                            if not prod.empty:
                                st.write(f"• {prod.iloc[0]['title']}")
                else:
                    st.error("Hybrid model not available")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <p style="text-align: center; color: #666; font-size: 0.9rem;">
            Amazon Electronics Recommendation System | Hybrid AI Model
        </p>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
