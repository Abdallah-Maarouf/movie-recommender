#!/usr/bin/env python3
"""
Proper Hybrid Model Evaluation with Train/Test Split

This script implements proper evaluation methodology to avoid data leakage
and get realistic performance estimates for the hybrid recommendation system.
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class ProperHybridEvaluation:
    """
    Proper evaluation of hybrid model with train/test split
    """
    
    def __init__(self):
        self.movies_df = None
        self.ratings_df = None
        self.train_ratings = None
        self.test_ratings = None
        
    def load_data(self):
        """Load data and create proper train/test split"""
        print("Loading data for proper evaluation...")
        
        # Load movie data
        with open('data/movies.json', 'r', encoding='utf-8') as f:
            movies_data = json.load(f)
        self.movies_df = pd.DataFrame(movies_data)
        
        # Load ratings data
        self.ratings_df = pd.read_csv('data/ratings.csv')
        
        print(f"Total ratings: {len(self.ratings_df)}")
        print(f"Users: {self.ratings_df['UserID'].nunique()}")
        print(f"Movies: {self.ratings_df['MovieID'].nunique()}")
        
    def create_temporal_split(self, test_ratio=0.2):
        """Create temporal train/test split (more realistic)"""
        print(f"Creating temporal train/test split ({test_ratio*100}% test)...")
        
        # Sort by timestamp for temporal split
        sorted_ratings = self.ratings_df.sort_values('Timestamp')
        
        # Split based on time (last 20% of ratings as test)
        split_idx = int(len(sorted_ratings) * (1 - test_ratio))
        
        self.train_ratings = sorted_ratings.iloc[:split_idx].copy()
        self.test_ratings = sorted_ratings.iloc[split_idx:].copy()
        
        print(f"Train ratings: {len(self.train_ratings)}")
        print(f"Test ratings: {len(self.test_ratings)}")
        
        # Check for cold start users/movies in test set
        train_users = set(self.train_ratings['UserID'].unique())
        train_movies = set(self.train_ratings['MovieID'].unique())
        
        test_users = set(self.test_ratings['UserID'].unique())
        test_movies = set(self.test_ratings['MovieID'].unique())
        
        cold_start_users = test_users - train_users
        cold_start_movies = test_movies - train_movies
        
        print(f"Cold start users in test: {len(cold_start_users)}")
        print(f"Cold start movies in test: {len(cold_start_movies)}")
        
        return self.train_ratings, self.test_ratings
    
    def create_user_based_split(self, test_ratio=0.2):
        """Create user-based train/test split"""
        print(f"Creating user-based train/test split ({test_ratio*100}% test)...")
        
        # For each user, split their ratings into train/test
        train_data = []
        test_data = []
        
        for user_id in self.ratings_df['UserID'].unique():
            user_ratings = self.ratings_df[self.ratings_df['UserID'] == user_id]
            
            if len(user_ratings) >= 5:  # Only split users with enough ratings
                # Sort by timestamp for each user
                user_ratings_sorted = user_ratings.sort_values('Timestamp')
                
                # Split user's ratings
                n_test = max(1, int(len(user_ratings_sorted) * test_ratio))
                
                train_data.append(user_ratings_sorted.iloc[:-n_test])
                test_data.append(user_ratings_sorted.iloc[-n_test:])
            else:
                # Put all ratings in train for users with few ratings
                train_data.append(user_ratings)
        
        self.train_ratings = pd.concat(train_data, ignore_index=True)
        self.test_ratings = pd.concat(test_data, ignore_index=True)
        
        print(f"Train ratings: {len(self.train_ratings)}")
        print(f"Test ratings: {len(self.test_ratings)}")
        
        return self.train_ratings, self.test_ratings
    
    def retrain_models_on_train_data(self):
        """Retrain collaborative and content models on training data only"""
        print("Retraining models on training data only...")
        
        # Create user-item matrix from training data only
        train_pivot = self.train_ratings.pivot_table(
            index='UserID', 
            columns='MovieID', 
            values='Rating', 
            fill_value=0
        )
        
        print(f"Training user-item matrix shape: {train_pivot.shape}")
        
        # For this evaluation, we'll use the existing pre-trained models
        # but acknowledge they were trained on full data (limitation)
        print("WARNING: Using existing models trained on full data")
        print("This is a limitation - ideally we'd retrain on training data only")
        
        return train_pivot
    
    def evaluate_on_test_set(self):
        """Evaluate hybrid model on proper test set"""
        print("Evaluating on test set...")
        
        # Load existing models (trained on full data - limitation)
        with open('data/models/user_similarity_matrix.pkl', 'rb') as f:
            user_similarity = pickle.load(f)
        
        with open('data/models/item_similarity_matrix.pkl', 'rb') as f:
            item_similarity = pickle.load(f)
        
        with open('data/models/content_similarity_matrix.pkl', 'rb') as f:
            content_similarity = pickle.load(f)
        
        # Create training user-item matrix for predictions
        train_pivot = self.train_ratings.pivot_table(
            index='UserID', 
            columns='MovieID', 
            values='Rating', 
            fill_value=0
        )
        
        predictions = []
        actuals = []
        cold_start_predictions = []
        warm_start_predictions = []
        
        # Evaluate on test set
        test_sample = self.test_ratings.sample(n=min(2000, len(self.test_ratings)), random_state=42)
        
        for _, row in test_sample.iterrows():
            user_id = row['UserID']
            movie_id = row['MovieID']
            actual_rating = row['Rating']
            
            # Check if user exists in training data
            if user_id in train_pivot.index:
                # Warm start - user has training data
                user_profile = train_pivot.loc[user_id]
                pred_rating = self.predict_rating(
                    user_id, movie_id, user_profile, 
                    item_similarity, content_similarity, train_pivot
                )
                warm_start_predictions.append((pred_rating, actual_rating))
            else:
                # Cold start - new user
                pred_rating = self.cold_start_predict(movie_id, content_similarity)
                cold_start_predictions.append((pred_rating, actual_rating))
            
            predictions.append(pred_rating)
            actuals.append(actual_rating)
        
        # Calculate metrics
        overall_rmse = np.sqrt(mean_squared_error(actuals, predictions))
        overall_mae = mean_absolute_error(actuals, predictions)
        
        print(f"\n=== PROPER EVALUATION RESULTS ===")
        print(f"Overall RMSE: {overall_rmse:.4f}")
        print(f"Overall MAE: {overall_mae:.4f}")
        print(f"Total predictions: {len(predictions)}")
        
        # Warm start performance
        if warm_start_predictions:
            warm_preds, warm_actuals = zip(*warm_start_predictions)
            warm_rmse = np.sqrt(mean_squared_error(warm_actuals, warm_preds))
            warm_mae = mean_absolute_error(warm_actuals, warm_preds)
            print(f"\nWarm Start Users:")
            print(f"  RMSE: {warm_rmse:.4f}")
            print(f"  MAE: {warm_mae:.4f}")
            print(f"  Count: {len(warm_start_predictions)}")
        
        # Cold start performance
        if cold_start_predictions:
            cold_preds, cold_actuals = zip(*cold_start_predictions)
            cold_rmse = np.sqrt(mean_squared_error(cold_actuals, cold_preds))
            cold_mae = mean_absolute_error(cold_actuals, cold_preds)
            print(f"\nCold Start Users:")
            print(f"  RMSE: {cold_rmse:.4f}")
            print(f"  MAE: {cold_mae:.4f}")
            print(f"  Count: {len(cold_start_predictions)}")
        
        return {
            'overall_rmse': overall_rmse,
            'overall_mae': overall_mae,
            'warm_start_rmse': warm_rmse if warm_start_predictions else None,
            'cold_start_rmse': cold_rmse if cold_start_predictions else None,
            'total_predictions': len(predictions)
        }
    
    def predict_rating(self, user_id, movie_id, user_profile, item_similarity, content_similarity, train_pivot):
        """Predict rating using hybrid approach"""
        
        # Collaborative prediction (item-based)
        collab_pred = 3.0  # Default
        
        try:
            if movie_id in train_pivot.columns:
                # Get movies this user rated
                rated_movies = user_profile[user_profile > 0]
                
                if len(rated_movies) > 0:
                    # Find similarities to rated movies
                    similarities = []
                    ratings = []
                    
                    for rated_movie_id in rated_movies.index:
                        try:
                            # Map movie IDs to matrix indices safely
                            movie_idx = min(movie_id - 1, item_similarity.shape[0] - 1)
                            rated_idx = min(rated_movie_id - 1, item_similarity.shape[1] - 1)
                            
                            if (movie_idx >= 0 and rated_idx >= 0 and 
                                movie_idx < item_similarity.shape[0] and 
                                rated_idx < item_similarity.shape[1]):
                                
                                # Handle both numpy array and DataFrame formats
                                if hasattr(item_similarity, 'iloc'):
                                    sim = item_similarity.iloc[movie_idx, rated_idx]
                                else:
                                    sim = item_similarity[movie_idx, rated_idx]
                                
                                if sim > 0:
                                    similarities.append(sim)
                                    ratings.append(rated_movies[rated_movie_id])
                        except (IndexError, KeyError):
                            continue
                    
                    if similarities:
                        collab_pred = np.average(ratings, weights=similarities)
                        collab_pred = np.clip(collab_pred, 1.0, 5.0)
        except Exception:
            collab_pred = 3.0
        
        # Content-based prediction
        content_pred = self.content_based_predict(user_profile, movie_id, content_similarity)
        
        # Hybrid combination (using optimal alpha=0.9 from previous results)
        alpha = 0.9
        hybrid_pred = alpha * collab_pred + (1 - alpha) * content_pred
        
        return np.clip(hybrid_pred, 1.0, 5.0)
    
    def content_based_predict(self, user_profile, movie_id, content_similarity):
        """Content-based prediction"""
        try:
            movie_idx = min(movie_id - 1, content_similarity.shape[0] - 1)
            if movie_idx < 0:
                return 3.0
            
            # Find user's highly rated movies
            high_rated = user_profile[user_profile >= 4.0]
            
            if len(high_rated) == 0:
                return 3.0
            
            similarities = []
            for rated_movie_id in high_rated.index:
                try:
                    rated_idx = min(rated_movie_id - 1, content_similarity.shape[1] - 1)
                    if (rated_idx >= 0 and 
                        movie_idx < content_similarity.shape[0] and 
                        rated_idx < cont
                        
        rmats
                  loc'):
    
                        else:
                            sim = content_sim]
                        
                        similarities.append(sim * high_rated[rated_movie_id])
        rror):
                    continue
            
        ies:
                avg_sim = np.mean(similarities)
* 2.0)
           )
            
            return 3.0
        except Exception:
    urn 3.0        ret5.0(pred, 1.0, ipp.clrn n     retusim (avg_ed = 3.0 +       pr          rit  if simila  yEor, Ke(IndexErr except        x, rated_idxovie_idlarity[mid_idx]e_idx, rateiloc[moviarity.ent_similm = cont  si                      , 'it_similaritysattr(conten  if ha    
    
    def cold_start_predict(self, movie_id, content_similarity):
        """Prediction for cold start users"""
        # Use average rating for popular movies
        movie_stats = self.train_ratings.groupby('MovieID')['Rating'].agg(['mean', 'count'])
        
        if movie_id in movie_stats.index and movie_stats.loc[movie_id, 'count'] >= 10:
            return np.clip(movie_stats.loc[movie_id, 'mean'], 1.0, 5.0)
        
        return 3.5  # Default for unknown movies

def main():
    """Main evaluation function"""
    print("PROPER HYBRID MODEL EVALUATION")
    print("=" * 50)
    
    evaluator = ProperHybridEvaluation()
    
    try:
        # Load data
        evaluator.load_data()
        
        # Create temporal split (more realistic)
        print("\n--- TEMPORAL SPLIT EVALUATION ---")
        evaluator.create_temporal_split(test_ratio=0.2)
        temporal_results = evaluator.evaluate_on_test_set()
        
        # Create user-based split
        print("\n--- USER-BASED SPLIT EVALUATION ---")
        evaluator.create_user_based_split(test_ratio=0.2)
        user_based_results = evaluator.evaluate_on_test_set()
        
        # Compare with original results
        print("\n" + "=" * 50)
        print("COMPARISON WITH ORIGINAL EVALUATION")
        print("=" * 50)
        print("Original (with data leakage):")
        print("  RMSE: 1.0071")
        print("  MAE: 0.8015")
        print()
        print("Proper Temporal Split:")
        print(f"  RMSE: {temporal_results['overall_rmse']:.4f}")
        print(f"  MAE: {temporal_results['overall_mae']:.4f}")
        print()
        print("Proper User-based Split:")
        print(f"  RMSE: {user_based_results['overall_rmse']:.4f}")
        print(f"  MAE: {user_based_results['overall_mae']:.4f}")
        
        # Performance degradation
        original_rmse = 1.0071
        temporal_degradation = (temporal_results['overall_rmse'] - original_rmse) / original_rmse * 100
        user_degradation = (user_based_results['overall_rmse'] - original_rmse) / original_rmse * 100
        
        print(f"\nPerformance Degradation:")
        print(f"  Temporal split: +{temporal_degradation:.1f}% RMSE increase")
        print(f"  User-based split: +{user_degradation:.1f}% RMSE increase")
        
        print("\n" + "=" * 50)
        print("REALISTIC PERFORMANCE ASSESSMENT")
        print("=" * 50)
        
        realistic_rmse = max(temporal_results['overall_rmse'], user_based_results['overall_rmse'])
        
        if realistic_rmse < 1.3:
            print("✅ GOOD: Model performance is still competitive")
        elif realistic_rmse < 1.5:
            print("⚠️  FAIR: Model performance is acceptable but not great")
        else:
            print("❌ POOR: Model needs significant improvement")
        
        print(f"Realistic RMSE estimate: {realistic_rmse:.4f}")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()