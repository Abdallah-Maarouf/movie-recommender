#!/usr/bin/env python3
"""
Content-Based Filtering Model Training Script for Movie Recommendation System

This script implements content-based filtering using movie features (genres, year, metadata)
and TF-IDF similarity. It's designed to run in Google Colab with comprehensive evaluation
and model export functionality.

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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class ContentBasedFilteringTrainer:
    """
    Comprehensive content-based filtering model trainer using movie features and TF-IDF similarity.
    """
    
    def __init__(self, data_path: str = "./", models_path: str = "models/"):
        """
        Initialize the trainer with data and model paths.
        
        Args:
            data_path: Path to the data directory
            models_path: Path to save trained models
        """
        self.data_path = Path(data_path)
        self.models_path = Path(models_path)
        self.models_path.mkdir(parents=True, exist_ok=True)
        
        # Model parameters
        self.tfidf_max_features = 1000  # Maximum TF-IDF features
        self.min_df = 2  # Minimum document frequency
        self.max_df = 0.8  # Maximum document frequency
        
        # Data containers
        self.movies_df = None
        self.ratings_df = None
        self.movie_features = None
        self.content_similarity_matrix = None
        self.tfidf_vectorizer = None
        self.scaler = None
        
        # Evaluation metrics
        self.metrics = {}
        
    def load_data(self) -> None:
        """Load and preprocess the movie and ratings data."""
        logger.info("Loading movie and ratings data...")
        
        try:
            # Load movies data (from root directory)
            if Path("movies.json").exists():
                with open("movies.json", 'r') as f:
                    movies_data = json.load(f)
                self.movies_df = pd.DataFrame(movies_data)
                logger.info(f"Loaded {len(self.movies_df)} movies from movies.json")
            else:
                raise FileNotFoundError("Could not find movies.json in current directory")
            
            # Load ratings data (from root directory)
            if Path("ratings.csv").exists():
                self.ratings_df = pd.read_csv("ratings.csv")
                logger.info(f"Loaded {len(self.ratings_df)} ratings from ratings.csv")
            else:
                raise FileNotFoundError("Could not find ratings.csv in current directory")
                
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def extract_movie_features(self) -> None:
        """Extract and engineer features from movie metadata."""
        logger.info("Extracting movie features...")
        
        # Create a copy for feature engineering
        features_df = self.movies_df.copy()
        
        # 1. Genre Features - Create TF-IDF from genres
        logger.info("Processing genre features...")
        genre_text = features_df['Genres'].fillna('').str.replace('|', ' ')
        
        # 2. Year Features - Normalize and create decade groups
        logger.info("Processing year features...")
        features_df['Year_Normalized'] = (features_df['Year'] - features_df['Year'].min()) / (features_df['Year'].max() - features_df['Year'].min())
        features_df['Decade'] = (features_df['Year'] // 10) * 10
        
        # Create Era feature with proper handling of edge cases
        era_labels = []
        for year in features_df['Year']:
            if year < 1960:
                era_labels.append('Classic')
            elif year < 1980:
                era_labels.append('Retro')
            elif year < 1990:
                era_labels.append('Eighties')
            elif year < 2000:
                era_labels.append('Nineties')
            else:
                era_labels.append('Modern')
        
        features_df['Era'] = era_labels
        
        # 3. Calculate movie popularity and average ratings
        logger.info("Calculating popularity and rating statistics...")
        movie_stats = self.ratings_df.groupby('MovieID').agg({
            'Rating': ['mean', 'count', 'std'],
            'UserID': 'nunique'
        }).round(3)
        
        movie_stats.columns = ['AvgRating', 'NumRatings', 'RatingStd', 'NumUsers']
        movie_stats = movie_stats.fillna(0)
        
        # Merge with features
        features_df = features_df.merge(movie_stats, left_on='MovieID', right_index=True, how='left')
        
        # Fill NaN values carefully - handle different column types
        for col in features_df.columns:
            if features_df[col].dtype == 'object' or features_df[col].dtype.name == 'category':
                # For text/categorical columns, fill with empty string or most common value
                if col == 'Era':
                    features_df[col] = features_df[col].fillna('Unknown')
                else:
                    features_df[col] = features_df[col].fillna('')
            else:
                # For numerical columns, fill with 0
                features_df[col] = features_df[col].fillna(0)
        
        # 4. Create popularity score (log-scaled)
        features_df['PopularityScore'] = np.log1p(features_df['NumRatings'])
        features_df['QualityScore'] = features_df['AvgRating'] * np.log1p(features_df['NumRatings'])
        
        # 5. Genre diversity (number of genres)
        features_df['GenreCount'] = features_df['GenreList'].apply(len)
        
        # 6. Title length features
        features_df['TitleLength'] = features_df['CleanTitle'].str.len()
        features_df['TitleWordCount'] = features_df['CleanTitle'].str.split().str.len()
        
        # Store processed features
        self.movie_features = features_df
        
        logger.info(f"Extracted features for {len(features_df)} movies")
        logger.info(f"Feature columns: {list(features_df.columns)}")
    
    def create_tfidf_features(self) -> None:
        """Create TF-IDF features from movie content."""
        logger.info("Creating TF-IDF features...")
        
        # Combine genres and era information for TF-IDF
        content_text = []
        for _, movie in self.movie_features.iterrows():
            # Create content string from genres and era
            genres = movie['Genres'].replace('|', ' ') if pd.notna(movie['Genres']) else ''
            era = str(movie['Era']) if pd.notna(movie['Era']) else ''
            decade = f"decade_{movie['Decade']}" if pd.notna(movie['Decade']) else ''
            
            # Combine all text features
            combined_text = f"{genres} {era} {decade}".strip()
            content_text.append(combined_text)
        
        # Create TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self.tfidf_max_features,
            min_df=self.min_df,
            max_df=self.max_df,
            stop_words='english',
            ngram_range=(1, 2)  # Include bigrams
        )
        
        # Fit and transform
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(content_text)
        
        # Convert to DataFrame for easier handling
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), 
                               index=self.movie_features['MovieID'], 
                               columns=feature_names)
        
        logger.info(f"Created TF-IDF matrix: {tfidf_matrix.shape}")
        logger.info(f"Top TF-IDF features: {list(feature_names[:10])}")
        
        return tfidf_df
    
    def create_numerical_features(self) -> pd.DataFrame:
        """Create and normalize numerical features."""
        logger.info("Creating numerical features...")
        
        # Select numerical features
        numerical_features = [
            'Year_Normalized', 'AvgRating', 'PopularityScore', 
            'QualityScore', 'GenreCount', 'TitleLength', 'TitleWordCount'
        ]
        
        # Extract numerical features
        num_features_df = self.movie_features[['MovieID'] + numerical_features].copy()
        
        # Normalize numerical features
        self.scaler = StandardScaler()
        scaled_features = self.scaler.fit_transform(num_features_df[numerical_features])
        
        # Create DataFrame
        scaled_df = pd.DataFrame(scaled_features, 
                                index=num_features_df['MovieID'], 
                                columns=[f"{col}_scaled" for col in numerical_features])
        
        logger.info(f"Created {len(numerical_features)} normalized numerical features")
        
        return scaled_df
    
    def combine_features(self) -> pd.DataFrame:
        """Combine TF-IDF and numerical features."""
        logger.info("Combining all features...")
        
        # Get TF-IDF features
        tfidf_features = self.create_tfidf_features()
        
        # Get numerical features
        numerical_features = self.create_numerical_features()
        
        # Combine features
        combined_features = pd.concat([tfidf_features, numerical_features], axis=1)
        
        logger.info(f"Combined feature matrix: {combined_features.shape}")
        
        return combined_features
    
    def compute_content_similarity(self) -> None:
        """Compute content-based similarity matrix."""
        logger.info("Computing content-based similarity matrix...")
        
        start_time = time.time()
        
        # Get combined features
        feature_matrix = self.combine_features()
        
        # Compute cosine similarity
        logger.info(f"Computing similarity for {len(feature_matrix)} movies...")
        similarity_matrix = cosine_similarity(feature_matrix.values)
        
        # Create DataFrame
        self.content_similarity_matrix = pd.DataFrame(
            similarity_matrix,
            index=feature_matrix.index,
            columns=feature_matrix.index
        )
        
        computation_time = time.time() - start_time
        logger.info(f"Content similarity computation completed in {computation_time:.2f} seconds")
        
        self.metrics['similarity_computation_time'] = computation_time
        self.metrics['similarity_matrix_shape'] = self.content_similarity_matrix.shape
    
    def generate_content_recommendations(self, movie_id: int, n_recommendations: int = 10) -> List[Tuple[int, float, str]]:
        """Generate content-based recommendations for a movie."""
        if movie_id not in self.content_similarity_matrix.index:
            return []
        
        # Get similarity scores for the movie
        similarities = self.content_similarity_matrix.loc[movie_id]
        
        # Sort by similarity (excluding the movie itself)
        similar_movies = similarities.drop(movie_id).sort_values(ascending=False)
        
        # Get top N recommendations with movie titles
        recommendations = []
        for similar_movie_id, similarity_score in similar_movies.head(n_recommendations).items():
            movie_info = self.movie_features[self.movie_features['MovieID'] == similar_movie_id].iloc[0]
            recommendations.append((similar_movie_id, similarity_score, movie_info['Title']))
        
        return recommendations
    
    def create_user_content_profile(self, user_id: int) -> Optional[np.ndarray]:
        """Create a content profile for a user based on their rated movies."""
        # Get user's ratings
        user_ratings = self.ratings_df[self.ratings_df['UserID'] == user_id]
        
        if len(user_ratings) == 0:
            return None
        
        # Get feature matrix
        feature_matrix = self.combine_features()
        
        # Create weighted profile based on ratings
        user_profile = np.zeros(feature_matrix.shape[1])
        total_weight = 0
        
        for _, rating_row in user_ratings.iterrows():
            movie_id = rating_row['MovieID']
            rating = rating_row['Rating']
            
            if movie_id in feature_matrix.index:
                # Weight by rating (higher ratings contribute more)
                weight = rating / 5.0  # Normalize to 0-1
                movie_features = feature_matrix.loc[movie_id].values
                user_profile += weight * movie_features
                total_weight += weight
        
        if total_weight > 0:
            user_profile /= total_weight
        
        return user_profile
    
    def recommend_for_user(self, user_id: int, n_recommendations: int = 10) -> List[Tuple[int, float, str]]:
        """Generate content-based recommendations for a user."""
        # Create user profile
        user_profile = self.create_user_content_profile(user_id)
        
        if user_profile is None:
            return []
        
        # Get feature matrix
        feature_matrix = self.combine_features()
        
        # Get movies user hasn't rated
        user_rated_movies = set(self.ratings_df[self.ratings_df['UserID'] == user_id]['MovieID'])
        unrated_movies = [mid for mid in feature_matrix.index if mid not in user_rated_movies]
        
        # Calculate similarities to user profile
        recommendations = []
        for movie_id in unrated_movies:
            movie_features = feature_matrix.loc[movie_id].values
            similarity = cosine_similarity([user_profile], [movie_features])[0][0]
            
            movie_info = self.movie_features[self.movie_features['MovieID'] == movie_id].iloc[0]
            recommendations.append((movie_id, similarity, movie_info['Title']))
        
        # Sort by similarity and return top N
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:n_recommendations]
    
    def evaluate_content_based_model(self) -> None:
        """Evaluate the content-based model."""
        logger.info("Evaluating content-based model...")
        
        # Sample users for evaluation
        sample_users = self.ratings_df['UserID'].unique()[:100]
        
        predictions = []
        actuals = []
        
        logger.info(f"Evaluating on {len(sample_users)} users...")
        
        for i, user_id in enumerate(sample_users):
            if i % 20 == 0:
                logger.info(f"Progress: {i}/{len(sample_users)} users")
            
            # Get user's test ratings
            user_ratings = self.ratings_df[self.ratings_df['UserID'] == user_id]
            
            if len(user_ratings) < 5:  # Skip users with too few ratings
                continue
            
            # Split user's ratings for evaluation
            train_ratings, test_ratings = train_test_split(user_ratings, test_size=0.3, random_state=42)
            
            # Create user profile from training ratings
            user_profile = self.create_user_content_profile(user_id)
            
            if user_profile is None:
                continue
            
            # Get feature matrix
            feature_matrix = self.combine_features()
            
            # Predict ratings for test movies
            for _, test_rating in test_ratings.iterrows():
                movie_id = test_rating['MovieID']
                actual_rating = test_rating['Rating']
                
                if movie_id in feature_matrix.index:
                    movie_features = feature_matrix.loc[movie_id].values
                    similarity = cosine_similarity([user_profile], [movie_features])[0][0]
                    
                    # Convert similarity to rating scale (1-5)
                    predicted_rating = 1 + (similarity * 4)  # Scale 0-1 similarity to 1-5 rating
                    
                    predictions.append(predicted_rating)
                    actuals.append(actual_rating)
        
        # Calculate metrics
        if len(predictions) > 0:
            rmse = np.sqrt(mean_squared_error(actuals, predictions))
            mae = mean_absolute_error(actuals, predictions)
            
            self.metrics['content_rmse'] = rmse
            self.metrics['content_mae'] = mae
            self.metrics['evaluation_samples'] = len(predictions)
            
            logger.info(f"Content-based Model Performance:")
            logger.info(f"  RMSE: {rmse:.4f}")
            logger.info(f"  MAE: {mae:.4f}")
            logger.info(f"  Evaluated on {len(predictions)} predictions")
        else:
            logger.warning("No predictions generated for evaluation")
    
    def calculate_diversity_metrics(self) -> None:
        """Calculate diversity and coverage metrics."""
        logger.info("Calculating diversity metrics...")
        
        # Genre diversity in recommendations
        sample_movies = self.movie_features['MovieID'].sample(n=min(100, len(self.movie_features)), random_state=42)
        
        all_recommended_genres = set()
        genre_distributions = []
        
        for movie_id in sample_movies:
            recommendations = self.generate_content_recommendations(movie_id, n_recommendations=10)
            
            # Get genres of recommended movies
            rec_genres = []
            for rec_movie_id, _, _ in recommendations:
                movie_info = self.movie_features[self.movie_features['MovieID'] == rec_movie_id]
                if not movie_info.empty:
                    genres = movie_info.iloc[0]['GenreList']
                    rec_genres.extend(genres)
                    all_recommended_genres.update(genres)
            
            # Calculate genre distribution for this set of recommendations
            if rec_genres:
                genre_counts = pd.Series(rec_genres).value_counts(normalize=True)
                genre_distributions.append(genre_counts)
        
        # Calculate overall diversity metrics
        total_genres = len(self.movie_features['GenreList'].explode().unique())
        covered_genres = len(all_recommended_genres)
        
        self.metrics['genre_coverage'] = covered_genres / total_genres
        self.metrics['total_genres_in_recommendations'] = covered_genres
        self.metrics['total_possible_genres'] = total_genres
        
        # Calculate average genre entropy (diversity)
        if genre_distributions:
            avg_entropy = np.mean([
                -sum(p * np.log2(p) for p in dist.values if p > 0) 
                for dist in genre_distributions
            ])
            self.metrics['average_genre_entropy'] = avg_entropy
        
        logger.info(f"Diversity Metrics:")
        logger.info(f"  Genre Coverage: {self.metrics['genre_coverage']:.4f}")
        logger.info(f"  Covered Genres: {covered_genres}/{total_genres}")
        if 'average_genre_entropy' in self.metrics:
            logger.info(f"  Average Genre Entropy: {self.metrics['average_genre_entropy']:.4f}")
    
    def generate_sample_recommendations(self) -> Dict:
        """Generate sample recommendations for validation."""
        logger.info("Generating sample recommendations...")
        
        # Sample a popular movie for movie-to-movie recommendations
        popular_movies = self.movie_features.nlargest(5, 'NumRatings')
        sample_movie = popular_movies.iloc[0]
        
        movie_recommendations = self.generate_content_recommendations(
            sample_movie['MovieID'], n_recommendations=10
        )
        
        # Sample a user for user-based recommendations
        active_users = self.ratings_df['UserID'].value_counts().head(10)
        sample_user = active_users.index[0]
        
        user_recommendations = self.recommend_for_user(sample_user, n_recommendations=10)
        
        sample_results = {
            'movie_based': {
                'source_movie': {
                    'id': sample_movie['MovieID'],
                    'title': sample_movie['Title'],
                    'genres': sample_movie['Genres'],
                    'year': sample_movie['Year']
                },
                'recommendations': [
                    {
                        'movie_id': rec[0],
                        'similarity': rec[1],
                        'title': rec[2]
                    } for rec in movie_recommendations
                ]
            },
            'user_based': {
                'user_id': sample_user,
                'user_rating_count': len(self.ratings_df[self.ratings_df['UserID'] == sample_user]),
                'recommendations': [
                    {
                        'movie_id': rec[0],
                        'similarity': rec[1],
                        'title': rec[2]
                    } for rec in user_recommendations
                ]
            }
        }
        
        return sample_results
    
    def create_visualizations(self) -> None:
        """Create content-based model visualizations."""
        logger.info("Creating visualizations...")
        
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Genre distribution
        all_genres = self.movie_features['GenreList'].explode()
        genre_counts = all_genres.value_counts().head(15)
        
        axes[0, 0].barh(range(len(genre_counts)), genre_counts.values)
        axes[0, 0].set_yticks(range(len(genre_counts)))
        axes[0, 0].set_yticklabels(genre_counts.index)
        axes[0, 0].set_title('Top 15 Genres Distribution')
        axes[0, 0].set_xlabel('Number of Movies')
        
        # 2. Year distribution
        axes[0, 1].hist(self.movie_features['Year'], bins=30, alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('Movies by Release Year')
        axes[0, 1].set_xlabel('Year')
        axes[0, 1].set_ylabel('Number of Movies')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Rating vs Popularity scatter
        axes[1, 0].scatter(self.movie_features['AvgRating'], 
                          self.movie_features['PopularityScore'], 
                          alpha=0.6, s=30)
        axes[1, 0].set_title('Average Rating vs Popularity')
        axes[1, 0].set_xlabel('Average Rating')
        axes[1, 0].set_ylabel('Popularity Score (log)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Content similarity heatmap (sample)
        sample_movies = self.content_similarity_matrix.iloc[:20, :20]
        im = axes[1, 1].imshow(sample_movies.values, cmap='Blues', aspect='auto')
        axes[1, 1].set_title('Content Similarity Matrix Sample (20x20)')
        axes[1, 1].set_xlabel('Movies')
        axes[1, 1].set_ylabel('Movies')
        plt.colorbar(im, ax=axes[1, 1])
        
        plt.tight_layout()
        plt.savefig(self.models_path / 'content_based_visualizations.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info("Visualizations saved to content_based_visualizations.png")
    
    def save_models(self) -> None:
        """Save all trained models and artifacts."""
        logger.info("Saving content-based models and artifacts...")
        
        # Save content similarity matrix
        with open(self.models_path / 'content_similarity_matrix.pkl', 'wb') as f:
            pickle.dump(self.content_similarity_matrix, f)
        
        # Save movie features
        with open(self.models_path / 'movie_features.pkl', 'wb') as f:
            pickle.dump(self.movie_features, f)
        
        # Save TF-IDF vectorizer
        with open(self.models_path / 'tfidf_vectorizer.pkl', 'wb') as f:
            pickle.dump(self.tfidf_vectorizer, f)
        
        # Save scaler
        with open(self.models_path / 'feature_scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save metrics
        with open(self.models_path / 'content_metrics.json', 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        logger.info("All content-based models and artifacts saved successfully")
    
    def train_complete_pipeline(self) -> None:
        """Execute the complete content-based training pipeline."""
        logger.info("Starting content-based filtering training pipeline...")
        
        start_time = time.time()
        
        # Load and process data
        self.load_data()
        self.extract_movie_features()
        
        # Compute content similarity
        self.compute_content_similarity()
        
        # Evaluate model
        self.evaluate_content_based_model()
        
        # Calculate diversity metrics
        self.calculate_diversity_metrics()
        
        # Generate sample recommendations
        sample_recs = self.generate_sample_recommendations()
        
        # Display sample results
        logger.info("Sample Movie-to-Movie Recommendations:")
        source_movie = sample_recs['movie_based']['source_movie']
        logger.info(f"For '{source_movie['title']}' ({source_movie['year']}):")
        for i, rec in enumerate(sample_recs['movie_based']['recommendations'][:5]):
            logger.info(f"  {i+1}. {rec['title']} (similarity: {rec['similarity']:.3f})")
        
        # Create visualizations
        self.create_visualizations()
        
        # Save models
        self.save_models()
        
        total_time = time.time() - start_time
        self.metrics['total_training_time'] = total_time
        
        logger.info(f"Content-based training pipeline completed in {total_time:.2f} seconds")
        logger.info("="*50)
        logger.info("CONTENT-BASED TRAINING SUMMARY")
        logger.info("="*50)
        logger.info(f"Movies processed: {len(self.movie_features)}")
        logger.info(f"Feature dimensions: {self.combine_features().shape[1]}")
        logger.info(f"Similarity matrix: {self.content_similarity_matrix.shape}")
        if 'content_rmse' in self.metrics:
            logger.info(f"Content RMSE: {self.metrics['content_rmse']:.4f}")
            logger.info(f"Content MAE: {self.metrics['content_mae']:.4f}")
        logger.info(f"Genre Coverage: {self.metrics['genre_coverage']:.4f}")
        logger.info("="*50)


def main():
    """Main execution function for Google Colab."""
    print("🎬 Movie Recommendation System - Content-Based Filtering Training")
    print("="*60)
    
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
    
    # Initialize trainer
    trainer = ContentBasedFilteringTrainer()
    
    # Run complete training pipeline
    try:
        trainer.train_complete_pipeline()
        
        print("\n✅ Content-based training completed successfully!")
        print("\nNext steps:")
        print("1. Review the content-based metrics and visualizations")
        print("2. Compare with collaborative filtering results")
        print("3. Test the sample recommendations")
        print("4. Proceed to hybrid model training (Task 2.3)")
        print("\nFiles generated in models/ folder:")
        print("- content_similarity_matrix.pkl")
        print("- movie_features.pkl")
        print("- tfidf_vectorizer.pkl")
        print("- feature_scaler.pkl")
        print("- content_metrics.json")
        print("- content_based_visualizations.png")
        
    except Exception as e:
        logger.error(f"Content-based training failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()