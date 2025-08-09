#!/usr/bin/env python3
"""
Collaborative Filtering Model Training Script for Movie Recommendation System

This script implements collaborative filtering using matrix factorization (SVD) and 
similarity-based approaches. It's designed to run in Google Colab with comprehensive
logging, evaluation metrics, and model export functionality.

Author: Movie Recommendation System
Date: 2025
"""

import pandas as pd
import numpy as np
import json
import pickle
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.sparse import csr_matrix
import logging

# Configure logging (simplified for Colab)
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class CollaborativeFilteringTrainer:
    """
    Comprehensive collaborative filtering model trainer with SVD and similarity-based approaches.
    """
    
    def __init__(self, data_path: str = "./", models_path: str = "./models/"):
        """
        Initialize the trainer with data and model paths.
        
        Args:
            data_path: Path to the data directory (default: current directory)
            models_path: Path to save trained models
        """
        self.data_path = Path(data_path)
        self.models_path = Path(models_path)
        self.models_path.mkdir(parents=True, exist_ok=True)
        
        # Model parameters (optimized for Colab)
        self.svd_components = 50   # Reduced for faster training
        self.min_ratings_per_user = 10  # Reduced threshold
        self.min_ratings_per_movie = 5   # Reduced threshold
        
        # Data containers
        self.ratings_df = None
        self.movies_df = None
        self.user_item_matrix = None
        self.user_item_sparse = None
        self.train_matrix = None
        self.test_matrix = None
        
        # Trained models
        self.svd_model = None
        self.user_similarity_matrix = None
        self.item_similarity_matrix = None
        
        # Evaluation metrics
        self.metrics = {}
        
    def load_data(self) -> None:
        """Load and preprocess the MovieLens dataset."""
        logger.info("Loading MovieLens dataset...")
        
        try:
            # Load ratings data (check multiple possible locations)
            ratings_files = ["ratings.csv", "data/ratings.csv", "./ratings.csv"]
            for ratings_file in ratings_files:
                if Path(ratings_file).exists():
                    self.ratings_df = pd.read_csv(ratings_file)
                    logger.info(f"Loaded {len(self.ratings_df)} ratings from {ratings_file}")
                    break
            else:
                raise FileNotFoundError("Could not find ratings.csv in any expected location")
            
            # Load movies data (check multiple possible locations)
            movies_files = ["movies.json", "data/movies.json", "./movies.json"]
            for movies_file in movies_files:
                if Path(movies_file).exists():
                    with open(movies_file, 'r') as f:
                        movies_data = json.load(f)
                    self.movies_df = pd.DataFrame(movies_data)
                    logger.info(f"Loaded {len(self.movies_df)} movies from {movies_file}")
                    break
            else:
                raise FileNotFoundError("Could not find movies.json in any expected location")
            
            # Load data summary for context (check multiple possible locations)
            summary_files = ["data_summary.json", "data/data_summary.json", "./data_summary.json"]
            for summary_file in summary_files:
                if Path(summary_file).exists():
                    with open(summary_file, 'r') as f:
                        self.data_summary = json.load(f)
                    logger.info(f"Loaded data summary from {summary_file}")
                    break
            else:
                logger.warning("Could not find data_summary.json - continuing without it")
                self.data_summary = {}
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def preprocess_data(self) -> None:
        """Preprocess the data for collaborative filtering."""
        logger.info("Preprocessing data for collaborative filtering...")
        
        # Filter users and movies with sufficient ratings
        user_counts = self.ratings_df['UserID'].value_counts()
        movie_counts = self.ratings_df['MovieID'].value_counts()
        
        valid_users = user_counts[user_counts >= self.min_ratings_per_user].index
        valid_movies = movie_counts[movie_counts >= self.min_ratings_per_movie].index
        
        # Filter the ratings dataframe
        self.ratings_df = self.ratings_df[
            (self.ratings_df['UserID'].isin(valid_users)) &
            (self.ratings_df['MovieID'].isin(valid_movies))
        ]
        
        logger.info(f"After filtering: {len(self.ratings_df)} ratings, "
                   f"{len(valid_users)} users, {len(valid_movies)} movies")
        
        # Create user-item matrix
        self.user_item_matrix = self.ratings_df.pivot_table(
            index='UserID', 
            columns='MovieID', 
            values='Rating', 
            fill_value=0
        )
        
        # Convert to sparse matrix for memory efficiency
        self.user_item_sparse = csr_matrix(self.user_item_matrix.values)
        
        logger.info(f"Created user-item matrix: {self.user_item_matrix.shape}")
        logger.info(f"Matrix sparsity: {(1 - self.user_item_sparse.nnz / np.prod(self.user_item_matrix.shape)):.4f}")
    
    def create_train_test_split(self, test_size: float = 0.2, random_state: int = 42) -> None:
        """Create train/test split for evaluation."""
        logger.info(f"Creating train/test split with test_size={test_size}")
        
        # Split ratings data
        train_ratings, test_ratings = train_test_split(
            self.ratings_df, 
            test_size=test_size, 
            random_state=random_state,
            stratify=self.ratings_df['UserID']  # Ensure each user has ratings in both sets
        )
        
        # Create train matrix
        self.train_matrix = train_ratings.pivot_table(
            index='UserID', 
            columns='MovieID', 
            values='Rating', 
            fill_value=0
        ).reindex(
            index=self.user_item_matrix.index,
            columns=self.user_item_matrix.columns,
            fill_value=0
        )
        
        # Create test matrix (only test ratings, rest are 0)
        self.test_matrix = test_ratings.pivot_table(
            index='UserID', 
            columns='MovieID', 
            values='Rating', 
            fill_value=0
        ).reindex(
            index=self.user_item_matrix.index,
            columns=self.user_item_matrix.columns,
            fill_value=0
        )
        
        logger.info(f"Train set: {len(train_ratings)} ratings")
        logger.info(f"Test set: {len(test_ratings)} ratings")
    
    def train_svd_model(self) -> None:
        """Train SVD model for matrix factorization."""
        logger.info(f"Training SVD model with {self.svd_components} components...")
        
        start_time = time.time()
        
        # Initialize and fit SVD model
        self.svd_model = TruncatedSVD(
            n_components=self.svd_components,
            random_state=42,
            algorithm='randomized'
        )
        
        # Fit on training data
        train_sparse = csr_matrix(self.train_matrix.values)
        self.svd_model.fit(train_sparse)
        
        training_time = time.time() - start_time
        logger.info(f"SVD training completed in {training_time:.2f} seconds")
        
        # Log explained variance
        explained_variance_ratio = self.svd_model.explained_variance_ratio_
        total_variance = np.sum(explained_variance_ratio)
        logger.info(f"SVD explained variance ratio: {total_variance:.4f}")
        
        self.metrics['svd_training_time'] = training_time
        self.metrics['svd_explained_variance'] = total_variance
    
    def compute_similarity_matrices(self) -> None:
        """Compute user-user and item-item similarity matrices."""
        logger.info("Computing similarity matrices...")
        
        start_time = time.time()
        
        # User-user similarity (cosine similarity between users) - CHUNKED for memory
        logger.info("Computing user-user similarity matrix...")
        print(f"Computing similarity for {self.train_matrix.shape[0]} users...")
        
        # Use sparse matrix for memory efficiency
        from scipy.sparse import csr_matrix
        train_sparse = csr_matrix(self.train_matrix.values)
        user_similarity = cosine_similarity(train_sparse)
        
        self.user_similarity_matrix = pd.DataFrame(
            user_similarity,
            index=self.train_matrix.index,
            columns=self.train_matrix.index
        )
        logger.info(f"✅ User similarity matrix: {self.user_similarity_matrix.shape}")
        
        # Item-item similarity (cosine similarity between movies)
        logger.info("Computing item-item similarity matrix...")
        print(f"Computing similarity for {self.train_matrix.shape[1]} movies...")
        
        item_similarity = cosine_similarity(train_sparse.T)
        self.item_similarity_matrix = pd.DataFrame(
            item_similarity,
            index=self.train_matrix.columns,
            columns=self.train_matrix.columns
        )
        logger.info(f"✅ Item similarity matrix: {self.item_similarity_matrix.shape}")
        
        similarity_time = time.time() - start_time
        logger.info(f"Similarity computation completed in {similarity_time:.2f} seconds")
        
        self.metrics['similarity_computation_time'] = similarity_time
    
    def predict_svd(self, user_id: int, movie_id: int) -> float:
        """Predict rating using SVD model."""
        try:
            user_idx = self.user_item_matrix.index.get_loc(user_id)
            movie_idx = self.user_item_matrix.columns.get_loc(movie_id)
            
            # Transform user vector and reconstruct rating
            user_vector = self.train_matrix.iloc[user_idx:user_idx+1].values
            user_transformed = self.svd_model.transform(csr_matrix(user_vector))
            reconstructed = self.svd_model.inverse_transform(user_transformed)
            
            predicted_rating = reconstructed[0, movie_idx]
            
            # Clip to valid rating range
            return np.clip(predicted_rating, 1, 5)
            
        except (KeyError, ValueError):
            # Return global average for unknown users/movies
            return self.ratings_df['Rating'].mean()
    
    def predict_user_based(self, user_id: int, movie_id: int, k: int = 50) -> float:
        """Predict rating using user-based collaborative filtering."""
        try:
            if user_id not in self.user_similarity_matrix.index:
                return self.ratings_df['Rating'].mean()
            
            # Get similar users
            user_similarities = self.user_similarity_matrix.loc[user_id]
            similar_users = user_similarities.nlargest(k + 1)[1:]  # Exclude self
            
            # Get ratings from similar users for this movie
            numerator = 0
            denominator = 0
            
            for similar_user, similarity in similar_users.items():
                if movie_id in self.train_matrix.columns:
                    rating = self.train_matrix.loc[similar_user, movie_id]
                    if rating > 0:  # User has rated this movie
                        numerator += similarity * rating
                        denominator += abs(similarity)
            
            if denominator == 0:
                return self.ratings_df['Rating'].mean()
            
            predicted_rating = numerator / denominator
            return np.clip(predicted_rating, 1, 5)
            
        except (KeyError, ValueError):
            return self.ratings_df['Rating'].mean()
    
    def predict_item_based(self, user_id: int, movie_id: int, k: int = 50) -> float:
        """Predict rating using item-based collaborative filtering."""
        try:
            if movie_id not in self.item_similarity_matrix.index:
                return self.ratings_df['Rating'].mean()
            
            # Get similar movies
            movie_similarities = self.item_similarity_matrix.loc[movie_id]
            similar_movies = movie_similarities.nlargest(k + 1)[1:]  # Exclude self
            
            # Get user's ratings for similar movies
            numerator = 0
            denominator = 0
            
            for similar_movie, similarity in similar_movies.items():
                if user_id in self.train_matrix.index:
                    rating = self.train_matrix.loc[user_id, similar_movie]
                    if rating > 0:  # User has rated this movie
                        numerator += similarity * rating
                        denominator += abs(similarity)
            
            if denominator == 0:
                return self.ratings_df['Rating'].mean()
            
            predicted_rating = numerator / denominator
            return np.clip(predicted_rating, 1, 5)
            
        except (KeyError, ValueError):
            return self.ratings_df['Rating'].mean()
    
    def evaluate_models(self) -> None:
        """Evaluate all collaborative filtering models."""
        logger.info("Evaluating collaborative filtering models...")
        
        # Get test ratings
        test_ratings = []
        for user_id in self.test_matrix.index:
            for movie_id in self.test_matrix.columns:
                actual_rating = self.test_matrix.loc[user_id, movie_id]
                if actual_rating > 0:
                    test_ratings.append((user_id, movie_id, actual_rating))
        
        logger.info(f"Evaluating on {len(test_ratings)} test ratings...")
        
        # Evaluate SVD model
        svd_predictions = []
        user_predictions = []
        item_predictions = []
        actual_ratings = []
        
        for i, (user_id, movie_id, actual_rating) in enumerate(test_ratings):
            if i % 1000 == 0:
                logger.info(f"Processed {i}/{len(test_ratings)} predictions")
            
            # SVD prediction
            svd_pred = self.predict_svd(user_id, movie_id)
            svd_predictions.append(svd_pred)
            
            # User-based prediction
            user_pred = self.predict_user_based(user_id, movie_id)
            user_predictions.append(user_pred)
            
            # Item-based prediction
            item_pred = self.predict_item_based(user_id, movie_id)
            item_predictions.append(item_pred)
            
            actual_ratings.append(actual_rating)
        
        # Calculate metrics
        self.metrics['svd_rmse'] = np.sqrt(mean_squared_error(actual_ratings, svd_predictions))
        self.metrics['svd_mae'] = mean_absolute_error(actual_ratings, svd_predictions)
        
        self.metrics['user_based_rmse'] = np.sqrt(mean_squared_error(actual_ratings, user_predictions))
        self.metrics['user_based_mae'] = mean_absolute_error(actual_ratings, user_predictions)
        
        self.metrics['item_based_rmse'] = np.sqrt(mean_squared_error(actual_ratings, item_predictions))
        self.metrics['item_based_mae'] = mean_absolute_error(actual_ratings, item_predictions)
        
        logger.info("Evaluation Results:")
        logger.info(f"SVD - RMSE: {self.metrics['svd_rmse']:.4f}, MAE: {self.metrics['svd_mae']:.4f}")
        logger.info(f"User-based - RMSE: {self.metrics['user_based_rmse']:.4f}, MAE: {self.metrics['user_based_mae']:.4f}")
        logger.info(f"Item-based - RMSE: {self.metrics['item_based_rmse']:.4f}, MAE: {self.metrics['item_based_mae']:.4f}")
    
    def generate_sample_recommendations(self, user_id: int = None, n_recommendations: int = 10) -> Dict:
        """Generate sample recommendations for validation."""
        if user_id is None:
            # Select a random user with sufficient ratings
            user_id = np.random.choice(self.user_item_matrix.index)
        
        logger.info(f"Generating sample recommendations for user {user_id}")
        
        # Get user's rated movies
        user_ratings = self.user_item_matrix.loc[user_id]
        rated_movies = user_ratings[user_ratings > 0].index.tolist()
        unrated_movies = user_ratings[user_ratings == 0].index.tolist()
        
        # Generate predictions for unrated movies
        recommendations = []
        for movie_id in unrated_movies[:100]:  # Limit for performance
            svd_pred = self.predict_svd(user_id, movie_id)
            user_pred = self.predict_user_based(user_id, movie_id)
            item_pred = self.predict_item_based(user_id, movie_id)
            
            # Get movie info
            movie_info = self.movies_df[self.movies_df['MovieID'] == movie_id].iloc[0]
            
            recommendations.append({
                'movie_id': movie_id,
                'title': movie_info['Title'],
                'genres': movie_info['Genres'],
                'svd_prediction': svd_pred,
                'user_based_prediction': user_pred,
                'item_based_prediction': item_pred,
                'average_prediction': (svd_pred + user_pred + item_pred) / 3
            })
        
        # Sort by average prediction
        recommendations.sort(key=lambda x: x['average_prediction'], reverse=True)
        
        sample_recommendations = {
            'user_id': user_id,
            'user_rated_movies': len(rated_movies),
            'top_recommendations': recommendations[:n_recommendations]
        }
        
        return sample_recommendations
    
    def create_visualizations(self) -> None:
        """Create training progress and evaluation visualizations."""
        logger.info("Creating visualizations...")
        
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. SVD Explained Variance
        explained_variance = self.svd_model.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        
        axes[0, 0].plot(range(1, len(explained_variance) + 1), cumulative_variance)
        axes[0, 0].set_title('SVD Cumulative Explained Variance')
        axes[0, 0].set_xlabel('Number of Components')
        axes[0, 0].set_ylabel('Cumulative Explained Variance Ratio')
        axes[0, 0].grid(True)
        
        # 2. Model Performance Comparison
        models = ['SVD', 'User-based', 'Item-based']
        rmse_scores = [
            self.metrics['svd_rmse'],
            self.metrics['user_based_rmse'],
            self.metrics['item_based_rmse']
        ]
        mae_scores = [
            self.metrics['svd_mae'],
            self.metrics['user_based_mae'],
            self.metrics['item_based_mae']
        ]
        
        x = np.arange(len(models))
        width = 0.35
        
        axes[0, 1].bar(x - width/2, rmse_scores, width, label='RMSE', alpha=0.8)
        axes[0, 1].bar(x + width/2, mae_scores, width, label='MAE', alpha=0.8)
        axes[0, 1].set_title('Model Performance Comparison')
        axes[0, 1].set_xlabel('Models')
        axes[0, 1].set_ylabel('Error')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(models)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Rating Distribution
        axes[1, 0].hist(self.ratings_df['Rating'], bins=5, alpha=0.7, edgecolor='black')
        axes[1, 0].set_title('Rating Distribution')
        axes[1, 0].set_xlabel('Rating')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. User-Item Matrix Sparsity Visualization (sample)
        sample_matrix = self.user_item_matrix.iloc[:50, :50].values
        im = axes[1, 1].imshow(sample_matrix, cmap='Blues', aspect='auto')
        axes[1, 1].set_title('User-Item Matrix Sample (50x50)')
        axes[1, 1].set_xlabel('Movies')
        axes[1, 1].set_ylabel('Users')
        plt.colorbar(im, ax=axes[1, 1])
        
        plt.tight_layout()
        plt.savefig(self.models_path / 'collaborative_training_visualizations.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info("Visualizations saved to collaborative_training_visualizations.png")
    
    def save_models(self) -> None:
        """Save all trained models and artifacts."""
        logger.info("Saving trained models and artifacts...")
        
        # Save SVD model
        with open(self.models_path / 'collaborative_svd_model.pkl', 'wb') as f:
            pickle.dump(self.svd_model, f)
        
        # Save similarity matrices
        with open(self.models_path / 'user_similarity_matrix.pkl', 'wb') as f:
            pickle.dump(self.user_similarity_matrix, f)
        
        with open(self.models_path / 'item_similarity_matrix.pkl', 'wb') as f:
            pickle.dump(self.item_similarity_matrix, f)
        
        # Save user-item matrix for inference
        with open(self.models_path / 'user_item_matrix.pkl', 'wb') as f:
            pickle.dump(self.user_item_matrix, f)
        
        # Save metrics
        with open(self.models_path / 'collaborative_metrics.json', 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        logger.info("All models and artifacts saved successfully")
    
    def train_complete_pipeline(self) -> None:
        """Execute the complete training pipeline."""
        logger.info("Starting collaborative filtering training pipeline...")
        
        start_time = time.time()
        
        # Load and preprocess data
        self.load_data()
        self.preprocess_data()
        
        # Create train/test split
        self.create_train_test_split()
        
        # Train models
        self.train_svd_model()
        self.compute_similarity_matrices()
        
        # Evaluate models
        self.evaluate_models()
        
        # Generate sample recommendations
        sample_recs = self.generate_sample_recommendations()
        logger.info("Sample Recommendations:")
        for i, rec in enumerate(sample_recs['top_recommendations'][:5]):
            logger.info(f"{i+1}. {rec['title']} (Avg Pred: {rec['average_prediction']:.2f})")
        
        # Create visualizations
        self.create_visualizations()
        
        # Save models
        self.save_models()
        
        total_time = time.time() - start_time
        self.metrics['total_training_time'] = total_time
        
        logger.info(f"Training pipeline completed in {total_time:.2f} seconds")
        logger.info("="*50)
        logger.info("TRAINING SUMMARY")
        logger.info("="*50)
        logger.info(f"Dataset: {len(self.ratings_df)} ratings, {len(self.user_item_matrix.index)} users, {len(self.user_item_matrix.columns)} movies")
        logger.info(f"SVD Components: {self.svd_components}")
        logger.info(f"Matrix Sparsity: {(1 - self.user_item_sparse.nnz / np.prod(self.user_item_matrix.shape)):.4f}")
        logger.info(f"Best Model: {'SVD' if self.metrics['svd_rmse'] < min(self.metrics['user_based_rmse'], self.metrics['item_based_rmse']) else 'Similarity-based'}")
        logger.info("="*50)


def main():
    """Main execution function for Google Colab."""
    print("🎬 Movie Recommendation System - Collaborative Filtering Training")
    print("="*60)
    
    # Check what files are available in current directory
    print("📁 Available files in current directory:")
    import os
    for file in os.listdir('.'):
        if file.endswith(('.csv', '.json', '.py')):
            print(f"  - {file}")
    print()
    
    # Initialize trainer (will look for files in current directory)
    trainer = CollaborativeFilteringTrainer()
    
    # Run complete training pipeline
    try:
        trainer.train_complete_pipeline()
        
        print("\n✅ Training completed successfully!")
        print("\nNext steps:")
        print("1. Review the training logs and metrics")
        print("2. Check the generated visualizations")
        print("3. Test the sample recommendations")
        print("4. Download the trained models from models/ folder")
        print("\nFiles generated in models/ folder:")
        print("- collaborative_svd_model.pkl")
        print("- user_similarity_matrix.pkl") 
        print("- item_similarity_matrix.pkl")
        print("- collaborative_metrics.json")
        print("- collaborative_training_visualizations.png")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()