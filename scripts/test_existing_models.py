#!/usr/bin/env python3
"""
Test script to explore and understand existing trained models
before creating the hybrid model training script.
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load all necessary data files"""
    print("Loading data files...")
    
    # Load movie data
    with open('data/movies.json', 'r', encoding='utf-8') as f:
        movies_data = json.load(f)
    movies_df = pd.DataFrame(movies_data)
    
    # Load ratings data
    ratings_df = pd.read_csv('data/ratings.csv')
    
    print(f"Movies: {len(movies_df)}")
    print(f"Ratings: {len(ratings_df)}")
    print(f"Users: {ratings_df['userId'].nunique()}")
    
    return movies_df, ratings_df

def test_collaborative_models():
    """Test and analyze collaborative filtering models"""
    print("\n" + "="*50)
    print("TESTING COLLABORATIVE FILTERING MODELS")
    print("="*50)
    
    # Load collaborative models
    with open('data/models/collaborative_svd_model.pkl', 'rb') as f:
        svd_model = pickle.load(f)
    
    with open('data/models/user_similarity_matrix.pkl', 'rb') as f:
        user_similarity = pickle.load(f)
    
    with open('data/models/item_similarity_matrix.pkl', 'rb') as f:
        item_similarity = pickle.load(f)
    
    with open('data/models/user_item_matrix.pkl', 'rb') as f:
        user_item_matrix = pickle.load(f)
    
    print(f"SVD Model type: {type(svd_model)}")
    print(f"SVD components: {svd_model.n_components}")
    print(f"User similarity matrix shape: {user_similarity.shape}")
    print(f"Item similarity matrix shape: {item_similarity.shape}")
    print(f"User-item matrix shape: {user_item_matrix.shape}")
    
    # Test SVD predictions
    print("\nTesting SVD predictions...")
    test_users = [1, 10, 100]
    test_movies = [1, 50, 100]
    
    for user_id in test_users:
        for movie_id in test_movies:
            if user_id < user_item_matrix.shape[0] and movie_id < user_item_matrix.shape[1]:
                # Get actual rating if exists
                actual = user_item_matrix[user_id, movie_id]
                
                # Get SVD prediction
                user_factors = svd_model.components_[:, user_id]
                item_factors = svd_model.components_[:, movie_id]
                svd_pred = np.dot(user_factors, item_factors)
                
                print(f"User {user_id}, Movie {movie_id}: Actual={actual:.2f}, SVD={svd_pred:.2f}")
    
    return {
        'svd_model': svd_model,
        'user_similarity': user_similarity,
        'item_similarity': item_similarity,
        'user_item_matrix': user_item_matrix
    }

def test_content_based_models():
    """Test and analyze content-based filtering models"""
    print("\n" + "="*50)
    print("TESTING CONTENT-BASED FILTERING MODELS")
    print("="*50)
    
    # Load content-based models
    with open('data/models/content_similarity_matrix.pkl', 'rb') as f:
        content_similarity = pickle.load(f)
    
    with open('data/models/movie_features.pkl', 'rb') as f:
        movie_features = pickle.load(f)
    
    with open('data/models/tfidf_vectorizer.pkl', 'rb') as f:
        tfidf_vectorizer = pickle.load(f)
    
    print(f"Content similarity matrix shape: {content_similarity.shape}")
    print(f"Movie features shape: {movie_features.shape}")
    print(f"TF-IDF vectorizer vocabulary size: {len(tfidf_vectorizer.vocabulary_)}")
    
    # Test content-based recommendations
    print("\nTesting content-based similarity...")
    test_movie_ids = [0, 10, 50, 100]
    
    for movie_id in test_movie_ids:
        if movie_id < content_similarity.shape[0]:
            # Get most similar movies
            similarities = content_similarity[movie_id]
            top_similar = np.argsort(similarities)[-6:-1][::-1]  # Top 5 excluding self
            
            print(f"\nMovie {movie_id} most similar movies:")
            for similar_id in top_similar:
                print(f"  Movie {similar_id}: similarity = {similarities[similar_id]:.3f}")
    
    return {
        'content_similarity': content_similarity,
        'movie_features': movie_features,
        'tfidf_vectorizer': tfidf_vectorizer
    }

def analyze_model_performance():
    """Analyze and compare model performance"""
    print("\n" + "="*50)
    print("ANALYZING MODEL PERFORMANCE")
    print("="*50)
    
    # Load metrics
    with open('data/models/collaborative_metrics.json', 'r') as f:
        collab_metrics = json.load(f)
    
    with open('data/models/content_metrics.json', 'r') as f:
        content_metrics = json.load(f)
    
    print("Collaborative Filtering Metrics:")
    for key, value in collab_metrics.items():
        print(f"  {key}: {value}")
    
    print("\nContent-Based Filtering Metrics:")
    for key, value in content_metrics.items():
        print(f"  {key}: {value}")
    
    # Compare RMSE and MAE
    print("\nPerformance Comparison:")
    print(f"Collaborative (Item-based) RMSE: {collab_metrics['item_based_rmse']:.4f}")
    print(f"Content-based RMSE: {content_metrics['content_rmse']:.4f}")
    print(f"Collaborative (Item-based) MAE: {collab_metrics['item_based_mae']:.4f}")
    print(f"Content-based MAE: {content_metrics['content_mae']:.4f}")
    
    return collab_metrics, content_metrics

def test_recommendation_generation(collab_models, content_models, movies_df, ratings_df):
    """Test generating recommendations from both models"""
    print("\n" + "="*50)
    print("TESTING RECOMMENDATION GENERATION")
    print("="*50)
    
    # Select a test user with sufficient ratings
    user_ratings = ratings_df[ratings_df['userId'] == 1].copy()
    print(f"Test user has {len(user_ratings)} ratings")
    
    if len(user_ratings) == 0:
        print("No ratings found for test user")
        return
    
    # Get user's rated movies
    rated_movies = set(user_ratings['movieId'].values)
    print(f"User rated movies: {list(rated_movies)[:10]}...")
    
    # Test collaborative filtering recommendations
    print("\nGenerating collaborative recommendations...")
    user_id = 0  # Assuming 0-indexed
    if user_id < collab_models['user_item_matrix'].shape[0]:
        user_ratings_vector = collab_models['user_item_matrix'][user_id]
        
        # Find unrated movies
        unrated_mask = user_ratings_vector == 0
        unrated_indices = np.where(unrated_mask)[0]
        
        if len(unrated_indices) > 0:
            # Use item-based collaborative filtering
            item_sim = collab_models['item_similarity']
            predictions = []
            
            for movie_idx in unrated_indices[:20]:  # Test first 20 unrated movies
                # Get similarity to rated movies
                rated_indices = np.where(user_ratings_vector > 0)[0]
                if len(rated_indices) > 0:
                    similarities = item_sim[movie_idx, rated_indices]
                    ratings = user_ratings_vector[rated_indices]
                    
                    # Weighted average prediction
                    if np.sum(np.abs(similarities)) > 0:
                        pred = np.sum(similarities * ratings) / np.sum(np.abs(similarities))
                        predictions.append((movie_idx, pred))
            
            # Sort by predicted rating
            predictions.sort(key=lambda x: x[1], reverse=True)
            print(f"Top 5 collaborative recommendations:")
            for movie_idx, pred_rating in predictions[:5]:
                print(f"  Movie {movie_idx}: predicted rating = {pred_rating:.2f}")
    
    # Test content-based recommendations
    print("\nGenerating content-based recommendations...")
    if len(user_ratings) > 0:
        # Get user's favorite movies (rating >= 4)
        favorite_movies = user_ratings[user_ratings['rating'] >= 4]['movieId'].values
        
        if len(favorite_movies) > 0:
            content_sim = content_models['content_similarity']
            
            # Find movies similar to user's favorites
            similar_movies = {}
            for fav_movie in favorite_movies[:5]:  # Use top 5 favorites
                if fav_movie < content_sim.shape[0]:
                    similarities = content_sim[fav_movie]
                    top_similar = np.argsort(similarities)[-11:-1][::-1]  # Top 10 excluding self
                    
                    for similar_movie in top_similar:
                        if similar_movie not in rated_movies:
                            if similar_movie not in similar_movies:
                                similar_movies[similar_movie] = []
                            similar_movies[similar_movie].append(similarities[similar_movie])
            
            # Average similarity scores
            content_recommendations = []
            for movie_id, sims in similar_movies.items():
                avg_sim = np.mean(sims)
                content_recommendations.append((movie_id, avg_sim))
            
            content_recommendations.sort(key=lambda x: x[1], reverse=True)
            print(f"Top 5 content-based recommendations:")
            for movie_idx, sim_score in content_recommendations[:5]:
                print(f"  Movie {movie_idx}: similarity score = {sim_score:.3f}")

def main():
    """Main testing function"""
    print("TESTING EXISTING TRAINED MODELS")
    print("="*60)
    
    try:
        # Load data
        movies_df, ratings_df = load_data()
        
        # Test collaborative models
        collab_models = test_collaborative_models()
        
        # Test content-based models
        content_models = test_content_based_models()
        
        # Analyze performance
        collab_metrics, content_metrics = analyze_model_performance()
        
        # Test recommendation generation
        test_recommendation_generation(collab_models, content_models, movies_df, ratings_df)
        
        print("\n" + "="*60)
        print("MODEL TESTING COMPLETED SUCCESSFULLY")
        print("="*60)
        
        # Summary for hybrid model design
        print("\nSUMMARY FOR HYBRID MODEL DESIGN:")
        print("-" * 40)
        print(f"• Collaborative filtering performs better (RMSE: {collab_metrics['item_based_rmse']:.4f})")
        print(f"• Content-based has higher RMSE ({content_metrics['content_rmse']:.4f}) but good for cold start")
        print(f"• User-item matrix shape: {collab_models['user_item_matrix'].shape}")
        print(f"• Content similarity matrix shape: {content_models['content_similarity'].shape}")
        print("• Both models are loaded and functional")
        print("• Ready to create hybrid approach combining both strengths")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()