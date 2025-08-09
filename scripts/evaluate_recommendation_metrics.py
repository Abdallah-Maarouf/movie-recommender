#!/usr/bin/env python3
"""
Comprehensive Recommendation System Evaluation Script

This script calculates proper recommendation metrics including Precision@K, Recall@K,
Hit Rate, and other classification metrics for the trained collaborative filtering models.
"""

import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class RecommendationEvaluator:
    """
    Comprehensive evaluator for recommendation system metrics.
    """
    
    def __init__(self, models_path: str = "models/", data_path: str = "./"):
        self.models_path = Path(models_path)
        self.data_path = Path(data_path)
        
        # Load models and data
        self.load_models()
        self.load_data()
        
        # Evaluation parameters
        self.k_values = [5, 10, 20]  # Top-K recommendations to evaluate
        self.relevance_threshold = 4.0  # Ratings >= 4 are considered "relevant"
        
    def load_models(self):
        """Load all trained models."""
        print("📦 Loading trained models...")
        
        # Load SVD model
        with open(self.models_path / 'collaborative_svd_model.pkl', 'rb') as f:
            self.svd_model = pickle.load(f)
        
        # Load similarity matrices
        with open(self.models_path / 'user_similarity_matrix.pkl', 'rb') as f:
            self.user_similarity_matrix = pickle.load(f)
            
        with open(self.models_path / 'item_similarity_matrix.pkl', 'rb') as f:
            self.item_similarity_matrix = pickle.load(f)
            
        # Load user-item matrix
        with open(self.models_path / 'user_item_matrix.pkl', 'rb') as f:
            self.user_item_matrix = pickle.load(f)
            
        print("✅ All models loaded successfully")
    
    def load_data(self):
        """Load original data for evaluation."""
        print("📊 Loading evaluation data...")
        
        # Load ratings (check multiple locations)
        ratings_files = ["ratings.csv", "data/ratings.csv"]
        for ratings_file in ratings_files:
            if Path(ratings_file).exists():
                self.ratings_df = pd.read_csv(ratings_file)
                break
        else:
            raise FileNotFoundError("Could not find ratings.csv")
        
        # Load movies (check multiple locations)
        movies_files = ["movies.json", "data/movies.json"]
        for movies_file in movies_files:
            if Path(movies_file).exists():
                with open(movies_file, 'r') as f:
                    movies_data = json.load(f)
                self.movies_df = pd.DataFrame(movies_data)
                break
        else:
            raise FileNotFoundError("Could not find movies.json")
        
        print(f"✅ Loaded {len(self.ratings_df)} ratings and {len(self.movies_df)} movies")
    
    def create_test_set(self, test_size: float = 0.2):
        """Create test set for evaluation."""
        print("🔀 Creating test set for evaluation...")
        
        # Filter to users and movies in our trained models
        valid_users = self.user_item_matrix.index
        valid_movies = self.user_item_matrix.columns
        
        test_ratings = self.ratings_df[
            (self.ratings_df['UserID'].isin(valid_users)) &
            (self.ratings_df['MovieID'].isin(valid_movies))
        ]
        
        # Sample test set
        n_test = int(len(test_ratings) * test_size)
        self.test_set = test_ratings.sample(n=n_test, random_state=42)
        
        print(f"✅ Created test set with {len(self.test_set)} ratings")
        
        # Create ground truth relevance
        self.ground_truth = {}
        for _, row in self.test_set.iterrows():
            user_id = row['UserID']
            movie_id = row['MovieID']
            rating = row['Rating']
            
            if user_id not in self.ground_truth:
                self.ground_truth[user_id] = {'relevant': set(), 'all_rated': set()}
            
            self.ground_truth[user_id]['all_rated'].add(movie_id)
            if rating >= self.relevance_threshold:
                self.ground_truth[user_id]['relevant'].add(movie_id)
        
        print(f"✅ Ground truth created for {len(self.ground_truth)} users")
    
    def predict_svd(self, user_id: int, movie_id: int) -> float:
        """Predict rating using SVD model."""
        try:
            user_idx = self.user_item_matrix.index.get_loc(user_id)
            movie_idx = self.user_item_matrix.columns.get_loc(movie_id)
            
            # Get user vector and transform
            user_vector = self.user_item_matrix.iloc[user_idx:user_idx+1].values
            user_transformed = self.svd_model.transform(user_vector)
            reconstructed = self.svd_model.inverse_transform(user_transformed)
            
            predicted_rating = reconstructed[0, movie_idx]
            return np.clip(predicted_rating, 1, 5)
            
        except (KeyError, ValueError):
            return 3.0  # Global average fallback
    
    def predict_user_based(self, user_id: int, movie_id: int, k: int = 50) -> float:
        """Predict rating using user-based collaborative filtering."""
        try:
            if user_id not in self.user_similarity_matrix.index:
                return 3.0
            
            # Get similar users
            user_similarities = self.user_similarity_matrix.loc[user_id]
            similar_users = user_similarities.nlargest(k + 1)[1:]  # Exclude self
            
            # Calculate weighted average
            numerator = 0
            denominator = 0
            
            for similar_user, similarity in similar_users.items():
                if movie_id in self.user_item_matrix.columns:
                    rating = self.user_item_matrix.loc[similar_user, movie_id]
                    if rating > 0:
                        numerator += similarity * rating
                        denominator += abs(similarity)
            
            if denominator == 0:
                return 3.0
            
            predicted_rating = numerator / denominator
            return np.clip(predicted_rating, 1, 5)
            
        except (KeyError, ValueError):
            return 3.0
    
    def predict_item_based(self, user_id: int, movie_id: int, k: int = 50) -> float:
        """Predict rating using item-based collaborative filtering."""
        try:
            if movie_id not in self.item_similarity_matrix.index:
                return 3.0
            
            # Get similar movies
            movie_similarities = self.item_similarity_matrix.loc[movie_id]
            similar_movies = movie_similarities.nlargest(k + 1)[1:]  # Exclude self
            
            # Calculate weighted average
            numerator = 0
            denominator = 0
            
            for similar_movie, similarity in similar_movies.items():
                if user_id in self.user_item_matrix.index:
                    rating = self.user_item_matrix.loc[user_id, similar_movie]
                    if rating > 0:
                        numerator += similarity * rating
                        denominator += abs(similarity)
            
            if denominator == 0:
                return 3.0
            
            predicted_rating = numerator / denominator
            return np.clip(predicted_rating, 1, 5)
            
        except (KeyError, ValueError):
            return 3.0
    
    def generate_recommendations(self, user_id: int, model_type: str, k: int = 20) -> List[Tuple[int, float]]:
        """Generate top-K recommendations for a user."""
        if user_id not in self.user_item_matrix.index:
            return []
        
        # Get movies user hasn't rated
        user_ratings = self.user_item_matrix.loc[user_id]
        unrated_movies = user_ratings[user_ratings == 0].index.tolist()
        
        # Predict ratings for unrated movies
        predictions = []
        for movie_id in unrated_movies:
            if model_type == 'svd':
                pred_rating = self.predict_svd(user_id, movie_id)
            elif model_type == 'user_based':
                pred_rating = self.predict_user_based(user_id, movie_id)
            elif model_type == 'item_based':
                pred_rating = self.predict_item_based(user_id, movie_id)
            else:
                continue
                
            predictions.append((movie_id, pred_rating))
        
        # Sort by predicted rating and return top-K
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:k]
    
    def calculate_precision_recall_at_k(self, user_id: int, recommendations: List[Tuple[int, float]], k: int) -> Tuple[float, float]:
        """Calculate Precision@K and Recall@K for a user."""
        if user_id not in self.ground_truth:
            return 0.0, 0.0
        
        # Get top-K recommendations
        top_k_items = [item_id for item_id, _ in recommendations[:k]]
        relevant_items = self.ground_truth[user_id]['relevant']
        
        if len(relevant_items) == 0:
            return 0.0, 0.0
        
        # Calculate metrics
        relevant_recommended = set(top_k_items) & relevant_items
        
        precision = len(relevant_recommended) / k if k > 0 else 0.0
        recall = len(relevant_recommended) / len(relevant_items) if len(relevant_items) > 0 else 0.0
        
        return precision, recall
    
    def calculate_hit_rate_at_k(self, user_id: int, recommendations: List[Tuple[int, float]], k: int) -> float:
        """Calculate Hit Rate@K for a user."""
        if user_id not in self.ground_truth:
            return 0.0
        
        top_k_items = set([item_id for item_id, _ in recommendations[:k]])
        relevant_items = self.ground_truth[user_id]['relevant']
        
        # Hit if at least one relevant item is in top-K
        return 1.0 if len(top_k_items & relevant_items) > 0 else 0.0
    
    def evaluate_model(self, model_type: str) -> Dict:
        """Evaluate a specific model with comprehensive metrics."""
        print(f"🔍 Evaluating {model_type} model...")
        
        results = {
            'model': model_type,
            'precision_at_k': {},
            'recall_at_k': {},
            'f1_at_k': {},
            'hit_rate_at_k': {},
            'coverage': 0.0,
            'diversity': 0.0
        }
        
        # Sample users for evaluation (to speed up computation)
        test_users = list(self.ground_truth.keys())[:100]  # Evaluate on 100 users
        print(f"Evaluating on {len(test_users)} users...")
        
        all_recommendations = []
        
        for k in self.k_values:
            precisions = []
            recalls = []
            hit_rates = []
            
            for i, user_id in enumerate(test_users):
                if i % 20 == 0:
                    print(f"  Progress: {i}/{len(test_users)} users")
                
                # Generate recommendations
                recommendations = self.generate_recommendations(user_id, model_type, k=20)
                all_recommendations.extend([rec[0] for rec in recommendations[:k]])
                
                # Calculate metrics
                precision, recall = self.calculate_precision_recall_at_k(user_id, recommendations, k)
                hit_rate = self.calculate_hit_rate_at_k(user_id, recommendations, k)
                
                precisions.append(precision)
                recalls.append(recall)
                hit_rates.append(hit_rate)
            
            # Average metrics
            avg_precision = np.mean(precisions)
            avg_recall = np.mean(recalls)
            avg_hit_rate = np.mean(hit_rates)
            
            # F1 score
            f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0.0
            
            results['precision_at_k'][k] = avg_precision
            results['recall_at_k'][k] = avg_recall
            results['f1_at_k'][k] = f1_score
            results['hit_rate_at_k'][k] = avg_hit_rate
        
        # Calculate coverage (percentage of items recommended)
        unique_recommendations = set(all_recommendations)
        total_items = len(self.user_item_matrix.columns)
        results['coverage'] = len(unique_recommendations) / total_items
        
        print(f"✅ {model_type} evaluation completed")
        return results
    
    def run_comprehensive_evaluation(self) -> Dict:
        """Run comprehensive evaluation on all models."""
        print("🎯 Starting Comprehensive Recommendation Evaluation")
        print("=" * 60)
        
        # Create test set
        self.create_test_set()
        
        # Evaluate all models
        models_to_evaluate = ['svd', 'user_based', 'item_based']
        all_results = {}
        
        for model_type in models_to_evaluate:
            all_results[model_type] = self.evaluate_model(model_type)
        
        return all_results
    
    def print_results(self, results: Dict):
        """Print comprehensive results."""
        print("\n" + "=" * 60)
        print("📊 COMPREHENSIVE RECOMMENDATION EVALUATION RESULTS")
        print("=" * 60)
        
        for model_name, model_results in results.items():
            print(f"\n🎯 {model_name.upper()} MODEL:")
            print("-" * 40)
            
            for k in self.k_values:
                precision = model_results['precision_at_k'][k]
                recall = model_results['recall_at_k'][k]
                f1 = model_results['f1_at_k'][k]
                hit_rate = model_results['hit_rate_at_k'][k]
                
                print(f"📈 Top-{k} Metrics:")
                print(f"   Precision@{k}: {precision:.4f}")
                print(f"   Recall@{k}:    {recall:.4f}")
                print(f"   F1-Score@{k}:  {f1:.4f}")
                print(f"   Hit Rate@{k}:  {hit_rate:.4f}")
                print()
            
            print(f"📊 Coverage: {model_results['coverage']:.4f}")
            print()
        
        # Find best model for each metric
        print("🏆 BEST MODELS BY METRIC:")
        print("-" * 30)
        
        for k in self.k_values:
            best_precision = max(results.keys(), key=lambda x: results[x]['precision_at_k'][k])
            best_recall = max(results.keys(), key=lambda x: results[x]['recall_at_k'][k])
            best_f1 = max(results.keys(), key=lambda x: results[x]['f1_at_k'][k])
            best_hit_rate = max(results.keys(), key=lambda x: results[x]['hit_rate_at_k'][k])
            
            print(f"Top-{k}:")
            print(f"  Best Precision: {best_precision} ({results[best_precision]['precision_at_k'][k]:.4f})")
            print(f"  Best Recall:    {best_recall} ({results[best_recall]['recall_at_k'][k]:.4f})")
            print(f"  Best F1-Score:  {best_f1} ({results[best_f1]['f1_at_k'][k]:.4f})")
            print(f"  Best Hit Rate:  {best_hit_rate} ({results[best_hit_rate]['hit_rate_at_k'][k]:.4f})")
            print()
    
    def save_results(self, results: Dict):
        """Save evaluation results."""
        output_file = self.models_path / 'recommendation_evaluation_metrics.json'
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Deep convert all values
        json_results = {}
        for model, model_results in results.items():
            json_results[model] = {}
            for key, value in model_results.items():
                if isinstance(value, dict):
                    json_results[model][key] = {k: convert_numpy(v) for k, v in value.items()}
                else:
                    json_results[model][key] = convert_numpy(value)
        
        with open(output_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"💾 Results saved to {output_file}")


def main():
    """Main execution function."""
    print("🎬 Comprehensive Recommendation System Evaluation")
    print("=" * 60)
    
    # Check available files
    print("📁 Available files:")
    import os
    for file in os.listdir('.'):
        if file.endswith(('.csv', '.json', '.pkl')):
            print(f"  - {file}")
    
    if os.path.exists('models'):
        print("📁 Models folder contents:")
        for file in os.listdir('models'):
            print(f"  - models/{file}")
    print()
    
    # Initialize evaluator
    evaluator = RecommendationEvaluator()
    
    # Run evaluation
    results = evaluator.run_comprehensive_evaluation()
    
    # Print and save results
    evaluator.print_results(results)
    evaluator.save_results(results)
    
    print("\n🎉 Comprehensive evaluation completed!")
    print("\nKey Files Generated:")
    print("- models/recommendation_evaluation_metrics.json")
    print("\nNow you have complete metrics including:")
    print("- Precision@K, Recall@K, F1-Score@K")
    print("- Hit Rate@K")
    print("- Coverage")
    print("- Model comparisons")


if __name__ == "__main__":
    main()