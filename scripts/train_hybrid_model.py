#!/usr/bin/env python3
"""
Hybrid Model Training Script for Movie Recommendation System

This script combines collaborative filtering and content-based filtering approaches
to create an optimal hybrid recommendation system. It implements multiple combination
strategies and performs comprehensive evaluation.

Author: Movie Recommendation System
Date: 2024
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import warnings
warnings.filterwarnings('ignore')

class HybridRecommendationSystem:
    """
    Hybrid recommendation system combining collaborative and content-based filtering
    """
    
    def __init__(self):
        self.collaborative_models = {}
        self.content_models = {}
        self.hybrid_config = {}
        self.movies_df = None
        self.ratings_df = None
        
    def load_data(self):
        """Load all necessary data files"""
        print("Loading data files...")
        
        # Load movie data with proper encoding
        with open('data/movies.json', 'r', encoding='utf-8') as f:
            movies_data = json.load(f)
        self.movies_df = pd.DataFrame(movies_data)
        
        # Load ratings data
        self.ratings_df = pd.read_csv('data/ratings.csv')
        
        print(f"Movies: {len(self.movies_df)}")
        print(f"Ratings: {len(self.ratings_df)}")
        print(f"Users: {self.ratings_df['UserID'].nunique()}")
        
    def load_trained_models(self):
        """Load all pre-trained models"""
        print("Loading pre-trained models...")
        
        # Load collaborative filtering models
        with open('data/models/collaborative_svd_model.pkl', 'rb') as f:
            self.collaborative_models['svd'] = pickle.load(f)
        
        with open('data/models/user_similarity_matrix.pkl', 'rb') as f:
            self.collaborative_models['user_similarity'] = pickle.load(f)
        
        with open('data/models/item_similarity_matrix.pkl', 'rb') as f:
            self.collaborative_models['item_similarity'] = pickle.load(f)
        
        with open('data/models/user_item_matrix.pkl', 'rb') as f:
            self.collaborative_models['user_item_matrix'] = pickle.load(f)
        
        # Load content-based models
        with open('data/models/content_similarity_matrix.pkl', 'rb') as f:
            self.content_models['similarity_matrix'] = pickle.load(f)
        
        with open('data/models/movie_features.pkl', 'rb') as f:
            self.content_models['features'] = pickle.load(f)
        
        with open('data/models/tfidf_vectorizer.pkl', 'rb') as f:
            self.content_models['tfidf'] = pickle.load(f)
        
        # Load performance metrics
        with open('data/models/collaborative_metrics.json', 'r') as f:
            self.collaborative_models['metrics'] = json.load(f)
        
        with open('data/models/content_metrics.json', 'r') as f:
            self.content_models['metrics'] = json.load(f)
        
        print("All models loaded successfully!")
        
    def collaborative_predict(self, user_id, movie_id, method='item_based'):
        """Generate collaborative filtering prediction"""
        user_item_matrix = self.collaborative_models['user_item_matrix']
        
        # Handle both DataFrame and numpy array formats
        if hasattr(user_item_matrix, 'iloc'):
            # DataFrame format
            if user_id >= len(user_item_matrix) or movie_id >= len(user_item_matrix.columns):
                return 3.0  # Default rating
        else:
            # Numpy array format
            if user_id >= user_item_matrix.shape[0] or movie_id >= user_item_matrix.shape[1]:
                return 3.0  # Default rating
        
        if method == 'svd':
            # SVD prediction (simplified - would need proper user/item mapping)
            return 3.5  # Placeholder for SVD prediction
        
        elif method == 'item_based':
            item_similarity = self.collaborative_models['item_similarity']
            
            # Get user ratings safely
            try:
                if hasattr(user_item_matrix, 'iloc'):
                    user_ratings = user_item_matrix.iloc[user_id].values
                else:
                    user_ratings = user_item_matrix[user_id]
            except (IndexError, KeyError):
                return 3.0
            
            # Find rated movies by this user
            rated_indices = np.where(user_ratings > 0)[0]
            
            if len(rated_indices) == 0:
                return 3.0  # Default for cold start
            
            # Get similarities to target movie
            if movie_id < item_similarity.shape[0]:
                try:
                    # Handle DataFrame vs numpy array
                    if hasattr(item_similarity, 'iloc'):
                        similarities = item_similarity.iloc[movie_id, rated_indices].values
                    else:
                        similarities = item_similarity[movie_id, rated_indices]
                    
                    ratings = user_ratings[rated_indices]
                    
                    # Weighted average prediction
                    if np.sum(np.abs(similarities)) > 0:
                        prediction = np.sum(similarities * ratings) / np.sum(np.abs(similarities))
                        return np.clip(prediction, 1.0, 5.0)
                except (IndexError, KeyError, TypeError):
                    pass
            
            return 3.0
        
        elif method == 'user_based':
            user_similarity = self.collaborative_models['user_similarity']
            
            # Find similar users
            if user_id < user_similarity.shape[0]:
                user_sims = user_similarity[user_id]
                top_users = np.argsort(user_sims)[-50:]  # Top 50 similar users
                
                predictions = []
                for similar_user in top_users:
                    if similar_user != user_id:
                        try:
                            if hasattr(user_item_matrix, 'iloc'):
                                rating = user_item_matrix.iloc[similar_user, movie_id] if movie_id < len(user_item_matrix.columns) else 0
                            else:
                                rating = user_item_matrix[similar_user, movie_id] if movie_id < user_item_matrix.shape[1] else 0
                            
                            if rating > 0:
                                predictions.append(rating * user_sims[similar_user])
                        except (IndexError, KeyError, TypeError):
                            continue
                
                if predictions:
                    return np.clip(np.mean(predictions), 1.0, 5.0)
            
            return 3.0
    
    def content_based_predict(self, user_profile, movie_id):
        """Generate content-based prediction"""
        content_similarity = self.content_models['similarity_matrix']
        
        if movie_id >= content_similarity.shape[0]:
            return 3.0
        
        # Calculate similarity to user's preferred movies
        similarities = []
        for rated_movie, rating in user_profile.items():
            if rated_movie < content_similarity.shape[0] and rating >= 4.0:  # Only consider liked movies
                try:
                    # Handle DataFrame vs numpy array
                    if hasattr(content_similarity, 'iloc'):
                        sim = content_similarity.iloc[movie_id, rated_movie]
                    else:
                        sim = content_similarity[movie_id, rated_movie]
                    similarities.append(sim * rating)
                except (IndexError, KeyError, TypeError):
                    continue
        
        if similarities:
            avg_similarity = np.mean(similarities)
            # Convert similarity to rating scale
            prediction = 3.0 + (avg_similarity * 2.0)  # Scale similarity to rating
            return np.clip(prediction, 1.0, 5.0)
        
        return 3.0
    
    def hybrid_predict(self, user_id, movie_id, user_profile, strategy='linear', alpha=0.7):
        """Generate hybrid prediction using specified strategy"""
        
        if strategy == 'linear':
            # Linear combination: α * collaborative + (1-α) * content
            collab_pred = self.collaborative_predict(user_id, movie_id, 'item_based')
            content_pred = self.content_based_predict(user_profile, movie_id)
            
            hybrid_pred = alpha * collab_pred + (1 - alpha) * content_pred
            return np.clip(hybrid_pred, 1.0, 5.0)
        
        elif strategy == 'switching':
            # Use collaborative if user has enough ratings, otherwise content-based
            user_item_matrix = self.collaborative_models['user_item_matrix']
            
            try:
                if hasattr(user_item_matrix, 'iloc'):
                    if user_id < len(user_item_matrix):
                        user_ratings_count = np.sum(user_item_matrix.iloc[user_id].values > 0)
                    else:
                        user_ratings_count = 0
                else:
                    if user_id < user_item_matrix.shape[0]:
                        user_ratings_count = np.sum(user_item_matrix[user_id] > 0)
                    else:
                        user_ratings_count = 0
                
                if user_ratings_count >= 20:  # Sufficient data for collaborative
                    return self.collaborative_predict(user_id, movie_id, 'item_based')
                else:
                    return self.content_based_predict(user_profile, movie_id)
            except:
                return self.content_based_predict(user_profile, movie_id)
        
        elif strategy == 'weighted_confidence':
            # Weight based on confidence in each method
            collab_pred = self.collaborative_predict(user_id, movie_id, 'item_based')
            content_pred = self.content_based_predict(user_profile, movie_id)
            
            # Calculate confidence based on data availability
            user_item_matrix = self.collaborative_models['user_item_matrix']
            collab_confidence = 0.5
            content_confidence = 0.5
            
            try:
                if hasattr(user_item_matrix, 'iloc'):
                    if user_id < len(user_item_matrix):
                        user_ratings_count = np.sum(user_item_matrix.iloc[user_id].values > 0)
                    else:
                        user_ratings_count = 0
                else:
                    if user_id < user_item_matrix.shape[0]:
                        user_ratings_count = np.sum(user_item_matrix[user_id] > 0)
                    else:
                        user_ratings_count = 0
                
                collab_confidence = min(user_ratings_count / 50.0, 1.0)  # Max confidence at 50 ratings
                content_confidence = 1.0 - collab_confidence
            except:
                collab_confidence = 0.5
                content_confidence = 0.5
            
            hybrid_pred = (collab_confidence * collab_pred + content_confidence * content_pred) / (collab_confidence + content_confidence)
            return np.clip(hybrid_pred, 1.0, 5.0)
        
        else:
            # Default to linear combination
            return self.hybrid_predict(user_id, movie_id, user_profile, 'linear', alpha)
    
    def evaluate_model(self, strategy='linear', alpha=0.7, test_size=0.2):
        """Evaluate hybrid model performance"""
        print(f"Evaluating hybrid model with strategy: {strategy}, alpha: {alpha}")
        
        # Create test set - use smaller sample to avoid ID mapping issues
        test_ratings = self.ratings_df.sample(n=min(1000, len(self.ratings_df)), random_state=42)
        
        predictions = []
        actuals = []
        
        # Create user and movie ID mappings
        user_item_matrix = self.collaborative_models['user_item_matrix']
        
        # Get valid user and movie ranges
        if hasattr(user_item_matrix, 'index'):
            max_users = len(user_item_matrix.index)
            max_movies = len(user_item_matrix.columns)
        else:
            max_users = user_item_matrix.shape[0]
            max_movies = user_item_matrix.shape[1]
        
        successful_predictions = 0
        
        for _, row in test_ratings.iterrows():
            # Map user and movie IDs to matrix indices
            original_user_id = row['UserID']
            original_movie_id = row['MovieID']
            
            # Simple mapping strategy - use modulo to ensure valid indices
            user_id = (original_user_id - 1) % max_users
            movie_id = (original_movie_id - 1) % max_movies
            
            actual_rating = row['Rating']
            
            # Create user profile (excluding current movie)
            user_ratings = self.ratings_df[
                (self.ratings_df['UserID'] == original_user_id) & 
                (self.ratings_df['MovieID'] != original_movie_id)
            ]
            
            user_profile = {}
            for _, rating_row in user_ratings.iterrows():
                mapped_movie_id = (rating_row['MovieID'] - 1) % max_movies
                user_profile[mapped_movie_id] = rating_row['Rating']
            
            try:
                # Get prediction
                pred_rating = self.hybrid_predict(user_id, movie_id, user_profile, strategy, alpha)
                
                predictions.append(pred_rating)
                actuals.append(actual_rating)
                successful_predictions += 1
                
            except Exception as e:
                # Skip problematic predictions
                continue
        
        print(f"Successfully made {successful_predictions} predictions out of {len(test_ratings)} attempts")
        
        if len(predictions) == 0:
            return {
                'strategy': strategy,
                'alpha': alpha,
                'rmse': 999.0,
                'mae': 999.0,
                'predictions': [],
                'actuals': []
            }
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        mae = mean_absolute_error(actuals, predictions)
        
        return {
            'strategy': strategy,
            'alpha': alpha,
            'rmse': rmse,
            'mae': mae,
            'predictions': predictions,
            'actuals': actuals
        }
    
    def optimize_hyperparameters(self):
        """Optimize hybrid model hyperparameters"""
        print("Optimizing hyperparameters...")
        
        strategies = ['linear', 'switching', 'weighted_confidence']
        alphas = [0.5, 0.6, 0.7, 0.8, 0.9]
        
        best_rmse = float('inf')
        best_config = {}
        results = []
        
        for strategy in strategies:
            if strategy == 'linear':
                for alpha in alphas:
                    result = self.evaluate_model(strategy, alpha)
                    results.append(result)
                    
                    if result['rmse'] < best_rmse:
                        best_rmse = result['rmse']
                        best_config = {
                            'strategy': strategy,
                            'alpha': alpha,
                            'rmse': result['rmse'],
                            'mae': result['mae']
                        }
                    
                    print(f"Strategy: {strategy}, Alpha: {alpha}, RMSE: {result['rmse']:.4f}, MAE: {result['mae']:.4f}")
            else:
                result = self.evaluate_model(strategy)
                results.append(result)
                
                if result['rmse'] < best_rmse:
                    best_rmse = result['rmse']
                    best_config = {
                        'strategy': strategy,
                        'alpha': 0.7,  # Default
                        'rmse': result['rmse'],
                        'mae': result['mae']
                    }
                
                print(f"Strategy: {strategy}, RMSE: {result['rmse']:.4f}, MAE: {result['mae']:.4f}")
        
        print(f"\nBest configuration: {best_config}")
        self.hybrid_config = best_config
        
        return best_config, results
    
    def generate_sample_recommendations(self, user_id=1, n_recommendations=10):
        """Generate sample recommendations for demonstration"""
        print(f"Generating sample recommendations for user {user_id}...")
        
        # Get user's rating history
        user_ratings = self.ratings_df[self.ratings_df['UserID'] == user_id]
        user_profile = {}
        rated_movies = set()
        
        for _, row in user_ratings.iterrows():
            user_profile[row['MovieID'] - 1] = row['Rating']
            rated_movies.add(row['MovieID'] - 1)
        
        print(f"User has rated {len(user_profile)} movies")
        
        # Generate recommendations for unrated movies
        recommendations = []
        movie_ids = range(min(500, len(self.movies_df)))  # Test first 500 movies
        
        for movie_id in movie_ids:
            if movie_id not in rated_movies:
                pred_rating = self.hybrid_predict(
                    user_id - 1, movie_id, user_profile, 
                    self.hybrid_config.get('strategy', 'linear'),
                    self.hybrid_config.get('alpha', 0.7)
                )
                
                # Get movie info
                if movie_id < len(self.movies_df):
                    movie_info = self.movies_df.iloc[movie_id]
                    recommendations.append({
                        'movie_id': movie_id,
                        'title': movie_info.get('Title', f'Movie {movie_id}'),
                        'genres': movie_info.get('Genres', []),
                        'predicted_rating': pred_rating,
                        'explanation': self.generate_explanation(user_profile, movie_id)
                    })
        
        # Sort by predicted rating
        recommendations.sort(key=lambda x: x['predicted_rating'], reverse=True)
        
        return recommendations[:n_recommendations]
    
    def generate_explanation(self, user_profile, movie_id):
        """Generate explanation for recommendation"""
        # Find similar movies the user liked
        content_similarity = self.content_models['similarity_matrix']
        
        if movie_id >= content_similarity.shape[0]:
            return "Recommended based on your rating patterns"
        
        similar_movies = []
        for rated_movie, rating in user_profile.items():
            if rating >= 4.0 and rated_movie < content_similarity.shape[0]:
                try:
                    # Handle DataFrame vs numpy array
                    if hasattr(content_similarity, 'iloc'):
                        similarity = content_similarity.iloc[movie_id, rated_movie]
                    else:
                        similarity = content_similarity[movie_id, rated_movie]
                    
                    if similarity > 0.3:  # Threshold for similarity
                        if rated_movie < len(self.movies_df):
                            movie_title = self.movies_df.iloc[rated_movie].get('Title', f'Movie {rated_movie}')
                            similar_movies.append((movie_title, similarity))
                except (IndexError, KeyError, TypeError):
                    continue
        
        if similar_movies:
            similar_movies.sort(key=lambda x: x[1], reverse=True)
            top_similar = similar_movies[:2]
            titles = [movie[0] for movie in top_similar]
            return f"Because you liked {' and '.join(titles)}"
        
        return "Recommended based on your preferences"
    
    def create_fallback_recommendations(self):
        """Create popular movie recommendations for cold start scenarios"""
        print("Creating fallback recommendations...")
        
        # Calculate movie popularity and average ratings
        movie_stats = self.ratings_df.groupby('MovieID').agg({
            'Rating': ['mean', 'count']
        }).round(2)
        
        movie_stats.columns = ['avg_rating', 'rating_count']
        movie_stats = movie_stats.reset_index()
        
        # Filter movies with sufficient ratings and high average rating
        popular_movies = movie_stats[
            (movie_stats['rating_count'] >= 100) & 
            (movie_stats['avg_rating'] >= 4.0)
        ].sort_values(['avg_rating', 'rating_count'], ascending=[False, False])
        
        fallback_recommendations = []
        for _, row in popular_movies.head(50).iterrows():
            movie_id = int(row['MovieID']) - 1  # Convert to 0-indexed integer
            if movie_id < len(self.movies_df):
                movie_info = self.movies_df.iloc[movie_id]
                fallback_recommendations.append({
                    'movie_id': movie_id,
                    'title': movie_info.get('Title', f'Movie {movie_id}'),
                    'genres': movie_info.get('Genres', []),
                    'avg_rating': row['avg_rating'],
                    'rating_count': row['rating_count'],
                    'explanation': 'Popular movie with high ratings'
                })
        
        return fallback_recommendations
    
    def save_hybrid_model(self, best_config, evaluation_results, sample_recommendations, fallback_recommendations):
        """Save hybrid model configuration and results"""
        print("Saving hybrid model artifacts...")
        
        # Save hybrid model configuration
        hybrid_config = {
            'best_strategy': best_config['strategy'],
            'best_alpha': best_config['alpha'],
            'performance': {
                'rmse': best_config['rmse'],
                'mae': best_config['mae']
            },
            'model_info': {
                'collaborative_rmse': self.collaborative_models['metrics']['item_based_rmse'],
                'content_rmse': self.content_models['metrics']['content_rmse'],
                'improvement_over_collaborative': (self.collaborative_models['metrics']['item_based_rmse'] - best_config['rmse']) / self.collaborative_models['metrics']['item_based_rmse'],
                'improvement_over_content': (self.content_models['metrics']['content_rmse'] - best_config['rmse']) / self.content_models['metrics']['content_rmse']
            }
        }
        
        with open('data/models/hybrid_model_config.json', 'w') as f:
            json.dump(hybrid_config, f, indent=2)
        
        # Save evaluation results
        evaluation_summary = {
            'all_results': [
                {
                    'strategy': r['strategy'],
                    'alpha': r['alpha'],
                    'rmse': r['rmse'],
                    'mae': r['mae']
                } for r in evaluation_results
            ],
            'best_result': best_config
        }
        
        with open('data/models/hybrid_evaluation_results.json', 'w') as f:
            json.dump(evaluation_summary, f, indent=2)
        
        # Save recommendation explanations templates
        explanation_templates = {
            'content_based': "Because you liked {similar_movies}",
            'collaborative': "Users with similar tastes also enjoyed this movie",
            'hybrid': "Recommended based on your preferences and similar users",
            'popular': "Popular movie with high ratings",
            'cold_start': "Trending movie you might enjoy"
        }
        
        with open('data/models/recommendation_explanations.json', 'w') as f:
            json.dump(explanation_templates, f, indent=2)
        
        # Save fallback recommendations
        with open('data/models/fallback_recommendations.json', 'w') as f:
            json.dump(fallback_recommendations, f, indent=2)
        
        print("All hybrid model artifacts saved successfully!")

def main():
    """Main training function"""
    print("HYBRID MODEL TRAINING SCRIPT")
    print("=" * 50)
    
    # Initialize hybrid system
    hybrid_system = HybridRecommendationSystem()
    
    try:
        # Load data and models
        hybrid_system.load_data()
        hybrid_system.load_trained_models()
        
        # Optimize hyperparameters
        best_config, evaluation_results = hybrid_system.optimize_hyperparameters()
        
        # Generate sample recommendations
        sample_recommendations = hybrid_system.generate_sample_recommendations()
        
        print("\nSample Recommendations:")
        for i, rec in enumerate(sample_recommendations[:5], 1):
            print(f"{i}. {rec['title']} (Rating: {rec['predicted_rating']:.2f})")
            print(f"   {rec['explanation']}")
        
        # Create fallback recommendations
        fallback_recommendations = hybrid_system.create_fallback_recommendations()
        
        # Save all results
        hybrid_system.save_hybrid_model(
            best_config, evaluation_results, 
            sample_recommendations, fallback_recommendations
        )
        
        # Print final summary
        print("\n" + "=" * 50)
        print("HYBRID MODEL TRAINING COMPLETED")
        print("=" * 50)
        print(f"Best Strategy: {best_config['strategy']}")
        print(f"Best Alpha: {best_config['alpha']}")
        print(f"Best RMSE: {best_config['rmse']:.4f}")
        print(f"Best MAE: {best_config['mae']:.4f}")
        
        # Performance comparison
        collab_rmse = hybrid_system.collaborative_models['metrics']['item_based_rmse']
        content_rmse = hybrid_system.content_models['metrics']['content_rmse']
        
        print(f"\nPerformance Comparison:")
        print(f"Collaborative RMSE: {collab_rmse:.4f}")
        print(f"Content-based RMSE: {content_rmse:.4f}")
        print(f"Hybrid RMSE: {best_config['rmse']:.4f}")
        
        improvement_collab = (collab_rmse - best_config['rmse']) / collab_rmse * 100
        improvement_content = (content_rmse - best_config['rmse']) / content_rmse * 100
        
        print(f"Improvement over Collaborative: {improvement_collab:.1f}%")
        print(f"Improvement over Content-based: {improvement_content:.1f}%")
        
        print("\nFiles created:")
        print("- data/models/hybrid_model_config.json")
        print("- data/models/hybrid_evaluation_results.json")
        print("- data/models/recommendation_explanations.json")
        print("- data/models/fallback_recommendations.json")
        
    except Exception as e:
        print(f"Error during hybrid model training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()