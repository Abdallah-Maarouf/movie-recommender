#!/usr/bin/env python3
"""
MovieLens Data Preprocessing Script

This script processes the MovieLens 1M dataset to prepare it for the movie recommendation system.
It cleans the data, extracts features, and creates curated datasets for both training and production use.
"""

import pandas as pd
import numpy as np
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

def load_movielens_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load MovieLens 1M dataset files."""
    print("Loading MovieLens 1M dataset...")
    
    # Load movies data
    movies = pd.read_csv(
        'ml-1m/movies.dat',
        sep='::',
        names=['MovieID', 'Title', 'Genres'],
        engine='python',
        encoding='latin-1'
    )
    
    # Load ratings data
    ratings = pd.read_csv(
        'ml-1m/ratings.dat',
        sep='::',
        names=['UserID', 'MovieID', 'Rating', 'Timestamp'],
        engine='python'
    )
    
    # Load users data
    users = pd.read_csv(
        'ml-1m/users.dat',
        sep='::',
        names=['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'],
        engine='python'
    )
    
    print(f"Loaded {len(movies)} movies, {len(ratings)} ratings, {len(users)} users")
    return movies, ratings, users

def clean_movie_data(movies: pd.DataFrame) -> pd.DataFrame:
    """Clean and process movie metadata."""
    print("Cleaning movie data...")
    
    movies_clean = movies.copy()
    
    # Extract year from title using regex
    def extract_year(title):
        match = re.search(r'\((\d{4})\)$', title)
        return int(match.group(1)) if match else None
    
    movies_clean['Year'] = movies_clean['Title'].apply(extract_year)
    
    # Clean title by removing year
    movies_clean['CleanTitle'] = movies_clean['Title'].apply(
        lambda x: re.sub(r'\s*\(\d{4}\)$', '', x).strip()
    )
    
    # Process genres
    movies_clean['GenreList'] = movies_clean['Genres'].apply(
        lambda x: [genre.strip() for genre in x.split('|')] if pd.notna(x) else []
    )
    
    # Handle special characters for web display
    movies_clean['WebTitle'] = movies_clean['CleanTitle'].apply(
        lambda x: x.replace('"', '&quot;').replace("'", '&#39;') if pd.notna(x) else x
    )
    
    print(f"Processed {len(movies_clean)} movies")
    return movies_clean

def analyze_data_quality(movies: pd.DataFrame, ratings: pd.DataFrame) -> Dict:
    """Analyze data quality and generate statistics."""
    print("Analyzing data quality...")
    
    # Rating statistics per movie
    rating_stats = ratings.groupby('MovieID').agg({
        'Rating': ['count', 'mean', 'std'],
        'UserID': 'nunique'
    }).round(3)
    
    rating_stats.columns = ['rating_count', 'avg_rating', 'rating_std', 'unique_users']
    
    # Merge with movie data
    movies_with_stats = movies.merge(rating_stats, on='MovieID', how='left')
    movies_with_stats['rating_count'] = movies_with_stats['rating_count'].fillna(0)
    
    # Data quality metrics
    quality_stats = {
        'total_movies': len(movies),
        'movies_with_ratings': len(movies_with_stats[movies_with_stats['rating_count'] > 0]),
        'movies_with_sufficient_ratings': len(movies_with_stats[movies_with_stats['rating_count'] >= 100]),
        'movies_with_high_ratings': len(movies_with_stats[movies_with_stats['rating_count'] >= 1000]),
        'avg_ratings_per_movie': movies_with_stats['rating_count'].mean(),
        'median_ratings_per_movie': movies_with_stats['rating_count'].median(),
        'rating_distribution': ratings['Rating'].value_counts().to_dict(),
        'genre_distribution': {},
        'year_range': {
            'min_year': int(movies['Year'].min()) if movies['Year'].notna().any() else None,
            'max_year': int(movies['Year'].max()) if movies['Year'].notna().any() else None
        }
    }
    
    # Genre distribution
    all_genres = []
    for genre_list in movies['GenreList']:
        all_genres.extend(genre_list)
    
    genre_counts = pd.Series(all_genres).value_counts()
    quality_stats['genre_distribution'] = genre_counts.to_dict()
    
    print(f"Quality analysis complete. {quality_stats['movies_with_high_ratings']} movies have 1000+ ratings")
    return quality_stats, movies_with_stats

def select_initial_movies(movies_with_stats: pd.DataFrame, target_count: int = 30) -> pd.DataFrame:
    """Select diverse movies for initial rating interface."""
    print(f"Selecting {target_count} diverse movies for initial rating interface...")
    
    # Filter movies with sufficient ratings (1000+)
    popular_movies = movies_with_stats[movies_with_stats['rating_count'] >= 1000].copy()
    
    if len(popular_movies) < target_count:
        print(f"Warning: Only {len(popular_movies)} movies have 1000+ ratings. Lowering threshold...")
        popular_movies = movies_with_stats[movies_with_stats['rating_count'] >= 500].copy()
    
    # Create genre balance
    selected_movies = []
    target_genres = ['Action', 'Drama', 'Comedy', 'Thriller', 'Romance', 'Sci-Fi', 'Horror', 'Adventure']
    movies_per_genre = max(1, target_count // len(target_genres))
    
    for genre in target_genres:
        genre_movies = popular_movies[
            popular_movies['GenreList'].apply(lambda x: genre in x)
        ].nlargest(movies_per_genre, 'rating_count')
        
        selected_movies.extend(genre_movies['MovieID'].tolist())
    
    # Fill remaining slots with highest-rated movies
    remaining_slots = target_count - len(selected_movies)
    if remaining_slots > 0:
        additional_movies = popular_movies[
            ~popular_movies['MovieID'].isin(selected_movies)
        ].nlargest(remaining_slots, 'rating_count')
        selected_movies.extend(additional_movies['MovieID'].tolist())
    
    # Get final selection
    initial_movies = movies_with_stats[
        movies_with_stats['MovieID'].isin(selected_movies[:target_count])
    ].copy()
    
    print(f"Selected {len(initial_movies)} movies covering {len(target_genres)} genres")
    return initial_movies

def export_processed_data(movies_clean: pd.DataFrame, ratings: pd.DataFrame, 
                         initial_movies: pd.DataFrame, quality_stats: Dict):
    """Export processed data to JSON and CSV files."""
    print("Exporting processed data...")
    
    # Ensure data directory exists
    Path('data').mkdir(exist_ok=True)
    
    # Export full movie metadata
    movies_export = movies_clean.copy()
    movies_export['GenreList'] = movies_export['GenreList'].apply(lambda x: x if isinstance(x, list) else [])
    
    movies_dict = movies_export.to_dict('records')
    with open('data/movies.json', 'w', encoding='utf-8') as f:
        json.dump(movies_dict, f, indent=2, ensure_ascii=False)
    
    # Export initial movies for rating interface
    initial_movies_export = initial_movies[[
        'MovieID', 'CleanTitle', 'WebTitle', 'Year', 'GenreList', 
        'rating_count', 'avg_rating'
    ]].copy()
    initial_movies_export['GenreList'] = initial_movies_export['GenreList'].apply(
        lambda x: x if isinstance(x, list) else []
    )
    
    initial_dict = initial_movies_export.to_dict('records')
    with open('data/initial_movies.json', 'w', encoding='utf-8') as f:
        json.dump(initial_dict, f, indent=2, ensure_ascii=False)
    
    # Export ratings for model training
    ratings.to_csv('data/ratings.csv', index=False)
    
    # Export data summary
    with open('data/data_summary.json', 'w', encoding='utf-8') as f:
        json.dump(quality_stats, f, indent=2, ensure_ascii=False)
    
    print("Data export complete!")
    print(f"- Full movie metadata: data/movies.json ({len(movies_dict)} movies)")
    print(f"- Initial movies: data/initial_movies.json ({len(initial_dict)} movies)")
    print(f"- Ratings matrix: data/ratings.csv ({len(ratings)} ratings)")
    print(f"- Data summary: data/data_summary.json")

def main():
    """Main processing pipeline."""
    print("Starting MovieLens data preprocessing...")
    
    try:
        # Load raw data
        movies, ratings, users = load_movielens_data()
        
        # Clean movie data
        movies_clean = clean_movie_data(movies)
        
        # Analyze data quality
        quality_stats, movies_with_stats = analyze_data_quality(movies_clean, ratings)
        
        # Select initial movies
        initial_movies = select_initial_movies(movies_with_stats)
        
        # Export processed data
        export_processed_data(movies_clean, ratings, initial_movies, quality_stats)
        
        print("\n✅ Data preprocessing completed successfully!")
        print("\nNext steps:")
        print("1. Review the generated data files in the 'data/' directory")
        print("2. Check data/data_summary.json for quality metrics")
        print("3. Verify initial_movies.json contains diverse movie selection")
        
    except Exception as e:
        print(f"❌ Error during preprocessing: {str(e)}")
        raise

if __name__ == "__main__":
    main()