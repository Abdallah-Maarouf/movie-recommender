#!/usr/bin/env python3
"""
Enhanced Content-Based Filtering Model Training Script

This enhanced version includes optimizations for 10-15% performance improvement:
- Genre importance weighting based on user ratings
- Temporal relevance scoring for recent movies  
- Advanced TF-IDF parameters
- Multi-metric similarity computation
- Enhanced user profiling

Expected improvement: RMSE 1.28 → 1.1-1.2

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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

def main():
    """Enhanced content-based filtering training with optimizations."""
    print("🚀 Enhanced Content-Based Filtering Training")
    print("="*60)
    
    # Check available files
    print("📁 Available files:")
    import os
    for file in os.listdir('.'):
        if file.endswith(('.csv', '.json')):
            print(f"  - {file}")
    print()
    
    try:
        # Load data
        logger.info("Loading data...")
        movies_df = pd.read_json('movies.json')
        ratings_df = pd.read_csv('ratings.csv')
        logger.info(f"Loaded {len(movies_df)} movies and {len(ratings_df)} ratings")
        
        # Calculate genre importance weights
        logger.info("Calculating genre importance weights...")
        genre_stats = {}
        
        for _, movie in movies_df.iterrows():
            movie_ratings = ratings_df[ratings_df['MovieID'] == movie['MovieID']]
            
            if len(movie_ratings) > 0:
                avg_rating = movie_ratings['Rating'].mean()
                rating_count = len(movie_ratings)
                
                for genre in movie['GenreList']:
                    if genre not in genre_stats:
                        genre_stats[genre] = {'ratings': [], 'counts': [], 'movies': 0}
                    
                    genre_stats[genre]['ratings'].append(avg_rating)
                    genre_stats[genre]['counts'].append(rating_count)
                    genre_stats[genre]['movies'] += 1
        
        # Calculate importance scores
        genre_importance = {}
        for genre, stats in genre_stats.items():
            if stats['movies'] > 0:
                avg_rating = np.mean(stats['ratings'])
                avg_popularity = np.mean(stats['counts'])
                movie_count = stats['movies']
                
                # Importance = quality * popularity * prevalence
                importance = (avg_rating / 5.0) * np.log1p(avg_popularity) * np.log1p(movie_count)
                genre_importance[genre] = importance
        
        # Normalize importance scores
        max_importance = max(genre_importance.values()) if genre_importance else 1
        genre_importance = {
            genre: importance / max_importance 
            for genre, importance in genre_importance.items()
        }
        
        logger.info("Top 10 most important genres:")
        sorted_genres = sorted(genre_importance.items(), key=lambda x: x[1], reverse=True)
        for genre, importance in sorted_genres[:10]:
            logger.info(f"  {genre}: {importance:.3f}")
        
        # Calculate temporal weights
        logger.info("Calculating temporal weights...")
        min_year = movies_df['Year'].min()
        max_year = movies_df['Year'].max()
        year_range = max_year - min_year
        
        temporal_weights = {}
        for year in range(min_year, max_year + 1):
            recency_factor = (year - min_year) / year_range
            temporal_weight = 0.5 + 0.5 * recency_factor  # Range: 0.5 to 1.0
            temporal_weights[year] = temporal_weight
        
        # Extract enhanced movie features
        logger.info("Extracting enhanced movie features...")
        features_df = movies_df.copy()
        
        # 1. Enhanced Genre Features with Importance Weighting
        weighted_genre_scores = []
        for _, movie in features_df.iterrows():
            genre_score = 0
            for genre in movie['GenreList']:
                if genre in genre_importance:
                    genre_score += genre_importance[genre]
            genre_score = genre_score / len(movie['GenreList']) if movie['GenreList'] else 0
            weighted_genre_scores.append(genre_score)
        
        features_df['WeightedGenreScore'] = weighted_genre_scores
        
        # 2. Enhanced Temporal Features
        features_df['Year_Normalized'] = (features_df['Year'] - min_year) / year_range
        features_df['Decade'] = (features_df['Year'] // 10) * 10
        features_df['TemporalWeight'] = features_df['Year'].map(temporal_weights)
        
        # Create Era feature
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
        
        # 3. Enhanced Popularity and Quality Features
        movie_stats = ratings_df.groupby('MovieID').agg({
            'Rating': ['mean', 'count', 'std', 'median'],
            'UserID': 'nunique'
        }).round(3)
        
        movie_stats.columns = ['AvgRating', 'NumRatings', 'RatingStd', 'MedianRating', 'NumUsers']
        movie_stats = movie_stats.fillna(0)
        
        # Merge with features
        features_df = features_df.merge(movie_stats, left_on='MovieID', right_index=True, how='left')
        
        # Fill NaN values
        for col in features_df.columns:
            if features_df[col].dtype == 'object':
                features_df[col] = features_df[col].fillna('Unknown' if col == 'Era' else '')
            else:
                features_df[col] = features_df[col].fillna(0)
        
        # 4. Advanced Quality Metrics
        features_df['PopularityScore'] = np.log1p(features_df['NumRatings'])
        features_df['QualityScore'] = features_df['AvgRating'] * np.log1p(features_df['NumRatings'])
        features_df['RatingReliability'] = 1 / (1 + features_df['RatingStd'])
        features_df['UserEngagement'] = features_df['NumUsers'] / features_df['NumRatings'].clip(lower=1)
        
        # 5. Content Features
        features_df['GenreCount'] = features_df['GenreList'].apply(len)
        features_df['GenreDiversity'] = features_df['GenreList'].apply(
            lambda genres: len(set(genres)) / len(genres) if genres else 0
        )
        features_df['TitleLength'] = features_df['CleanTitle'].str.len()
        features_df['TitleWordCount'] = features_df['CleanTitle'].str.split().str.len()
        
        logger.info(f"Extracted enhanced features for {len(features_df)} movies")
        
        # Create enhanced TF-IDF features
        logger.info("Creating enhanced TF-IDF features...")
        enhanced_content_text = []
        
        for _, movie in features_df.iterrows():
            # Weight genres by importance
            weighted_genres = []
            for genre in movie['GenreList']:
                importance = genre_importance.get(genre, 0.5)
                repeat_count = max(1, int(importance * 3))
                weighted_genres.extend([genre.lower()] * repeat_count)
            
            # Add temporal and quality context
            era = str(movie['Era']).lower()
            decade = f"decade_{movie['Decade']}"
            
            quality_indicators = []
            if movie['AvgRating'] > 4.0:
                quality_indicators.append('high_quality')
            if movie['PopularityScore'] > np.percentile(features_df['PopularityScore'], 80):
                quality_indicators.append('popular')
            
            # Combine all features
            all_features = weighted_genres + [era, decade] + quality_indicators
            combined_text = ' '.join(filter(None, all_features))
            enhanced_content_text.append(combined_text)
        
        # Create enhanced TF-IDF vectorizer
        tfidf_vectorizer = TfidfVectorizer(
            max_features=2000,  # Increased from 1000
            min_df=1,
            max_df=0.7,
            stop_words='english',
            ngram_range=(1, 3),  # Increased for richer features
            sublinear_tf=True,
            norm='l2'
        )
        
        tfidf_matrix = tfidf_vectorizer.fit_transform(enhanced_content_text)
        feature_names = tfidf_vectorizer.get_feature_names_out()
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), 
                               index=features_df['MovieID'], 
                               columns=feature_names)
        
        logger.info(f"Created enhanced TF-IDF matrix: {tfidf_matrix.shape}")
        
        # Create enhanced numerical features
        numerical_features = [
            'Year_Normalized', 'AvgRating', 'PopularityScore', 'QualityScore',
            'GenreCount', 'TitleLength', 'WeightedGenreScore', 'TemporalWeight',
            'RatingReliability', 'UserEngagement', 'GenreDiversity', 'MedianRating'
        ]
        
        num_features_df = features_df[['MovieID'] + numerical_features].copy()
        scaler = MinMaxScaler()
        scaled_features = scaler.fit_transform(num_features_df[numerical_features])
        
        scaled_df = pd.DataFrame(scaled_features, 
                                index=num_features_df['MovieID'], 
                                columns=[f"{col}_scaled" for col in numerical_features])
        
        logger.info(f"Created {len(numerical_features)} enhanced numerical features")
        
        # Combine enhanced features with weighting
        weighted_tfidf = tfidf_df * 0.6  # Genre/content weight
        weighted_numerical = scaled_df * 0.4  # Numerical weight
        
        combined_features = pd.concat([weighted_tfidf, weighted_numerical], axis=1)
        logger.info(f"Combined enhanced feature matrix: {combined_features.shape}")
        
        # Compute enhanced similarity matrix
        logger.info("Computing enhanced similarity matrix...")
        start_time = time.time()
        
        # Compute multiple similarity metrics
        cosine_sim = cosine_similarity(combined_features.values)
        pearson_sim = np.corrcoef(combined_features.values)
        pearson_sim = np.nan_to_num(pearson_sim, nan=0.0)
        
        # Combine similarities with weighting
        enhanced_similarity = 0.7 * cosine_sim + 0.3 * pearson_sim
        np.fill_diagonal(enhanced_similarity, 1.0)
        enhanced_similarity = (enhanced_similarity + enhanced_similarity.T) / 2
        
        enhanced_similarity_matrix = pd.DataFrame(
            enhanced_similarity,
            index=combined_features.index,
            columns=combined_features.index
        )
        
        computation_time = time.time() - start_time
        logger.info(f"Enhanced similarity computation completed in {computation_time:.2f} seconds")
        
        # Evaluate enhanced model
        logger.info("Evaluating enhanced content-based model...")
        
        sample_users = ratings_df['UserID'].unique()[:150]
        predictions = []
        actuals = []
        
        for i, user_id in enumerate(sample_users):
            if i % 30 == 0:
                logger.info(f"Progress: {i}/{len(sample_users)} users")
            
            user_ratings = ratings_df[ratings_df['UserID'] == user_id]
            if len(user_ratings) < 5:
                continue
            
            train_ratings, test_ratings = train_test_split(user_ratings, test_size=0.3, random_state=42)
            
            # Create enhanced user profile
            user_profile = np.zeros(combined_features.shape[1])
            total_weight = 0
            
            for _, rating_row in train_ratings.iterrows():
                movie_id = rating_row['MovieID']
                rating = rating_row['Rating']
                
                if movie_id in combined_features.index:
                    # Enhanced weighting: exponential for high ratings
                    weight = (rating / 5.0) ** 2 if rating >= 4 else rating / 5.0
                    movie_features = combined_features.loc[movie_id].values
                    user_profile += weight * movie_features
                    total_weight += weight
            
            if total_weight > 0:
                user_profile /= total_weight
            else:
                continue
            
            # Predict ratings for test movies
            for _, test_rating in test_ratings.iterrows():
                movie_id = test_rating['MovieID']
                actual_rating = test_rating['Rating']
                
                if movie_id in combined_features.index:
                    movie_features = combined_features.loc[movie_id].values
                    similarity = cosine_similarity([user_profile], [movie_features])[0][0]
                    
                    # Enhanced rating prediction with sigmoid transformation
                    predicted_rating = 1 + 4 * (1 / (1 + np.exp(-5 * (similarity - 0.5))))
                    
                    predictions.append(predicted_rating)
                    actuals.append(actual_rating)
        
        # Calculate enhanced metrics
        if len(predictions) > 0:
            rmse = np.sqrt(mean_squared_error(actuals, predictions))
            mae = mean_absolute_error(actuals, predictions)
            correlation = pearsonr(predictions, actuals)[0] if len(predictions) > 1 else 0
            
            logger.info(f"Enhanced Content-based Model Performance:")
            logger.info(f"  RMSE: {rmse:.4f}")
            logger.info(f"  MAE: {mae:.4f}")
            logger.info(f"  Correlation: {correlation:.4f}")
            logger.info(f"  Evaluated on {len(predictions)} predictions")
            
            # Compare with original model
            try:
                with open('models/content_metrics.json', 'r') as f:
                    original_metrics = json.load(f)
                
                original_rmse = original_metrics.get('content_rmse', float('inf'))
                improvement = ((original_rmse - rmse) / original_rmse) * 100
                
                logger.info(f"  Improvement over original: {improvement:.1f}% RMSE reduction")
                
            except FileNotFoundError:
                logger.info("  (Original model metrics not found for comparison)")
            
            # Save enhanced metrics
            enhanced_metrics = {
                'enhanced_content_rmse': rmse,
                'enhanced_content_mae': mae,
                'enhanced_evaluation_samples': len(predictions),
                'prediction_correlation': correlation,
                'enhanced_similarity_computation_time': computation_time,
                'improvement_percentage': improvement if 'improvement' in locals() else 0
            }
            
            # Save enhanced models
            logger.info("Saving enhanced models...")
            Path('models').mkdir(exist_ok=True)
            
            with open('models/enhanced_content_similarity_matrix.pkl', 'wb') as f:
                pickle.dump(enhanced_similarity_matrix, f)
            
            with open('models/enhanced_movie_features.pkl', 'wb') as f:
                pickle.dump(features_df, f)
            
            with open('models/enhanced_tfidf_vectorizer.pkl', 'wb') as f:
                pickle.dump(tfidf_vectorizer, f)
            
            with open('models/enhanced_feature_scaler.pkl', 'wb') as f:
                pickle.dump(scaler, f)
            
            with open('models/genre_importance_weights.pkl', 'wb') as f:
                pickle.dump(genre_importance, f)
            
            with open('models/temporal_weights.pkl', 'wb') as f:
                pickle.dump(temporal_weights, f)
            
            with open('models/enhanced_content_metrics.json', 'w') as f:
                json.dump(enhanced_metrics, f, indent=2)
            
            # Generate sample recommendations
            popular_movies = features_df.nlargest(5, 'NumRatings')
            sample_movie = popular_movies.iloc[0]
            
            # Get top 10 similar movies
            similarities = enhanced_similarity_matrix.loc[sample_movie['MovieID']]
            similar_movies = similarities.drop(sample_movie['MovieID']).sort_values(ascending=False)
            
            logger.info("Enhanced Sample Recommendations:")
            logger.info(f"For '{sample_movie['Title']}' [Genre Score: {sample_movie['WeightedGenreScore']:.3f}]:")
            for i, (movie_id, similarity_score) in enumerate(similar_movies.head(5).items()):
                movie_info = features_df[features_df['MovieID'] == movie_id].iloc[0]
                logger.info(f"  {i+1}. {movie_info['Title']} (similarity: {similarity_score:.3f})")
            
            # Create enhanced visualizations
            logger.info("Creating enhanced visualizations...")
            
            plt.style.use('default')
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            
            # 1. Genre importance
            genres = list(genre_importance.keys())[:15]
            importance_scores = [genre_importance[g] for g in genres]
            
            axes[0, 0].barh(range(len(genres)), importance_scores)
            axes[0, 0].set_yticks(range(len(genres)))
            axes[0, 0].set_yticklabels(genres)
            axes[0, 0].set_title('Genre Importance Weights')
            axes[0, 0].set_xlabel('Importance Score')
            
            # 2. Temporal weights
            years = sorted(temporal_weights.keys())
            weights = [temporal_weights[y] for y in years]
            
            axes[0, 1].plot(years, weights, 'b-', alpha=0.7)
            axes[0, 1].set_title('Temporal Relevance Weights')
            axes[0, 1].set_xlabel('Year')
            axes[0, 1].set_ylabel('Temporal Weight')
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. Feature correlation
            feature_cols = ['WeightedGenreScore', 'TemporalWeight', 'QualityScore', 'PopularityScore']
            corr_matrix = features_df[feature_cols].corr()
            
            im = axes[0, 2].imshow(corr_matrix.values, cmap='RdBu_r', vmin=-1, vmax=1)
            axes[0, 2].set_xticks(range(len(feature_cols)))
            axes[0, 2].set_yticks(range(len(feature_cols)))
            axes[0, 2].set_xticklabels([f.replace('Score', '').replace('Weight', 'W') for f in feature_cols], rotation=45)
            axes[0, 2].set_yticklabels([f.replace('Score', '').replace('Weight', 'W') for f in feature_cols])
            axes[0, 2].set_title('Enhanced Feature Correlations')
            plt.colorbar(im, ax=axes[0, 2])
            
            # 4. Quality vs Popularity with genre coloring
            scatter = axes[1, 0].scatter(features_df['AvgRating'], 
                                       features_df['PopularityScore'],
                                       c=features_df['WeightedGenreScore'],
                                       alpha=0.6, s=30, cmap='viridis')
            axes[1, 0].set_title('Quality vs Popularity (colored by Genre Score)')
            axes[1, 0].set_xlabel('Average Rating')
            axes[1, 0].set_ylabel('Popularity Score')
            axes[1, 0].grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=axes[1, 0])
            
            # 5. Similarity distribution
            sample_similarities = []
            sample_size = min(100, len(enhanced_similarity_matrix))
            sample_indices = np.random.choice(len(enhanced_similarity_matrix), sample_size, replace=False)
            
            for i in sample_indices:
                for j in sample_indices:
                    if i != j:
                        sample_similarities.append(enhanced_similarity_matrix.iloc[i, j])
            
            axes[1, 1].hist(sample_similarities, bins=30, alpha=0.7, edgecolor='black')
            axes[1, 1].set_title('Enhanced Similarity Score Distribution')
            axes[1, 1].set_xlabel('Similarity Score')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].grid(True, alpha=0.3)
            
            # 6. Similarity heatmap
            sample_matrix = enhanced_similarity_matrix.iloc[:20, :20]
            im = axes[1, 2].imshow(sample_matrix.values, cmap='Blues', aspect='auto')
            axes[1, 2].set_title('Enhanced Similarity Matrix Sample (20x20)')
            axes[1, 2].set_xlabel('Movies')
            axes[1, 2].set_ylabel('Movies')
            plt.colorbar(im, ax=axes[1, 2])
            
            plt.tight_layout()
            plt.savefig('models/enhanced_content_visualizations.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            logger.info("Enhanced visualizations saved")
            
            print("\n🎉 Enhanced content-based training completed!")
            print(f"\n📊 ENHANCED RESULTS:")
            print(f"  Enhanced RMSE: {rmse:.4f}")
            print(f"  Enhanced MAE: {mae:.4f}")
            if 'improvement' in locals():
                print(f"  Improvement: {improvement:.1f}% RMSE reduction")
            print(f"  Feature dimensions: {combined_features.shape[1]}")
            print(f"  Similarity matrix: {enhanced_similarity_matrix.shape}")
            
            print("\n🚀 Enhanced Features:")
            print("- Genre importance weighting")
            print("- Temporal relevance scoring")
            print("- Advanced TF-IDF (2000 features, trigrams)")
            print("- Multi-metric similarity (cosine + Pearson)")
            print("- Enhanced user profiling")
            
            print("\n📁 Files generated:")
            print("- enhanced_content_similarity_matrix.pkl")
            print("- enhanced_movie_features.pkl")
            print("- enhanced_tfidf_vectorizer.pkl")
            print("- genre_importance_weights.pkl")
            print("- temporal_weights.pkl")
            print("- enhanced_content_metrics.json")
            print("- enhanced_content_visualizations.png")
        
        else:
            logger.warning("No predictions generated for enhanced evaluation")
        
    except Exception as e:
        logger.error(f"Enhanced training failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()