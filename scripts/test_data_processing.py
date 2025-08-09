#!/usr/bin/env python3
"""
Data Processing Verification Test

This script verifies that the MovieLens data was properly processed and structured.
It performs various checks to ensure data quality and format correctness.
"""

import json
import pandas as pd
from pathlib import Path
import sys

def test_file_existence():
    """Test that all required output files exist."""
    print("🔍 Testing file existence...")
    
    required_files = [
        'data/movies.json',
        'data/initial_movies.json', 
        'data/ratings.csv',
        'data/data_summary.json'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    
    print("✅ All required files exist")
    return True

def test_movies_json():
    """Test the structure and content of movies.json."""
    print("\n🔍 Testing movies.json structure...")
    
    try:
        with open('data/movies.json', 'r', encoding='utf-8') as f:
            movies = json.load(f)
        
        # Check if it's a list
        if not isinstance(movies, list):
            print("❌ movies.json should contain a list")
            return False
        
        # Check minimum number of movies
        if len(movies) < 3000:
            print(f"❌ Expected at least 3000 movies, got {len(movies)}")
            return False
        
        # Check structure of first movie
        sample_movie = movies[0]
        required_fields = ['MovieID', 'CleanTitle', 'WebTitle', 'Year', 'GenreList']
        
        for field in required_fields:
            if field not in sample_movie:
                print(f"❌ Missing field '{field}' in movie data")
                return False
        
        # Check data types
        if not isinstance(sample_movie['MovieID'], int):
            print("❌ MovieID should be integer")
            return False
        
        if not isinstance(sample_movie['GenreList'], list):
            print("❌ GenreList should be a list")
            return False
        
        print(f"✅ movies.json contains {len(movies)} movies with correct structure")
        return True
        
    except Exception as e:
        print(f"❌ Error reading movies.json: {e}")
        return False

def test_initial_movies_json():
    """Test the structure and content of initial_movies.json."""
    print("\n🔍 Testing initial_movies.json structure...")
    
    try:
        with open('data/initial_movies.json', 'r', encoding='utf-8') as f:
            initial_movies = json.load(f)
        
        # Check if it's a list
        if not isinstance(initial_movies, list):
            print("❌ initial_movies.json should contain a list")
            return False
        
        # Check number of movies (should be around 30)
        if len(initial_movies) < 20 or len(initial_movies) > 35:
            print(f"❌ Expected 20-35 initial movies, got {len(initial_movies)}")
            return False
        
        # Check structure
        sample_movie = initial_movies[0]
        required_fields = ['MovieID', 'CleanTitle', 'WebTitle', 'Year', 'GenreList', 'rating_count', 'avg_rating']
        
        for field in required_fields:
            if field not in sample_movie:
                print(f"❌ Missing field '{field}' in initial movie data")
                return False
        
        # Check that movies have sufficient ratings
        low_rating_movies = [m for m in initial_movies if m['rating_count'] < 500]
        if len(low_rating_movies) > 5:  # Allow some flexibility
            print(f"❌ Too many movies with low rating counts: {len(low_rating_movies)}")
            return False
        
        # Check genre diversity
        all_genres = set()
        for movie in initial_movies:
            all_genres.update(movie['GenreList'])
        
        if len(all_genres) < 6:
            print(f"❌ Insufficient genre diversity: only {len(all_genres)} genres")
            return False
        
        print(f"✅ initial_movies.json contains {len(initial_movies)} movies with {len(all_genres)} genres")
        return True
        
    except Exception as e:
        print(f"❌ Error reading initial_movies.json: {e}")
        return False

def test_ratings_csv():
    """Test the structure and content of ratings.csv."""
    print("\n🔍 Testing ratings.csv structure...")
    
    try:
        ratings = pd.read_csv('data/ratings.csv')
        
        # Check columns
        expected_columns = ['UserID', 'MovieID', 'Rating', 'Timestamp']
        if list(ratings.columns) != expected_columns:
            print(f"❌ Expected columns {expected_columns}, got {list(ratings.columns)}")
            return False
        
        # Check minimum number of ratings
        if len(ratings) < 900000:
            print(f"❌ Expected at least 900,000 ratings, got {len(ratings)}")
            return False
        
        # Check rating range
        if ratings['Rating'].min() < 1 or ratings['Rating'].max() > 5:
            print(f"❌ Ratings should be 1-5, got range {ratings['Rating'].min()}-{ratings['Rating'].max()}")
            return False
        
        # Check for missing values
        if ratings.isnull().any().any():
            print("❌ Found missing values in ratings data")
            return False
        
        print(f"✅ ratings.csv contains {len(ratings)} ratings with correct structure")
        return True
        
    except Exception as e:
        print(f"❌ Error reading ratings.csv: {e}")
        return False

def test_data_summary_json():
    """Test the structure and content of data_summary.json."""
    print("\n🔍 Testing data_summary.json structure...")
    
    try:
        with open('data/data_summary.json', 'r', encoding='utf-8') as f:
            summary = json.load(f)
        
        # Check required fields
        required_fields = [
            'total_movies', 'movies_with_ratings', 'movies_with_sufficient_ratings',
            'rating_distribution', 'genre_distribution', 'year_range'
        ]
        
        for field in required_fields:
            if field not in summary:
                print(f"❌ Missing field '{field}' in data summary")
                return False
        
        # Check data reasonableness
        if summary['total_movies'] < 3000:
            print(f"❌ Expected at least 3000 total movies, got {summary['total_movies']}")
            return False
        
        if len(summary['genre_distribution']) < 10:
            print(f"❌ Expected at least 10 genres, got {len(summary['genre_distribution'])}")
            return False
        
        if summary['year_range']['min_year'] > 1920 or summary['year_range']['max_year'] < 2000:
            print(f"❌ Unexpected year range: {summary['year_range']}")
            return False
        
        print("✅ data_summary.json contains valid summary statistics")
        return True
        
    except Exception as e:
        print(f"❌ Error reading data_summary.json: {e}")
        return False

def test_data_consistency():
    """Test consistency between different data files."""
    print("\n🔍 Testing data consistency...")
    
    try:
        # Load all data
        with open('data/movies.json', 'r', encoding='utf-8') as f:
            movies = json.load(f)
        
        with open('data/initial_movies.json', 'r', encoding='utf-8') as f:
            initial_movies = json.load(f)
        
        ratings = pd.read_csv('data/ratings.csv')
        
        # Check that initial movies are subset of all movies
        all_movie_ids = {m['MovieID'] for m in movies}
        initial_movie_ids = {m['MovieID'] for m in initial_movies}
        
        if not initial_movie_ids.issubset(all_movie_ids):
            print("❌ Initial movies contain IDs not in main movie dataset")
            return False
        
        # Check that ratings reference valid movies
        rating_movie_ids = set(ratings['MovieID'].unique())
        invalid_ratings = rating_movie_ids - all_movie_ids
        
        if len(invalid_ratings) > 0:
            print(f"❌ Found {len(invalid_ratings)} ratings for non-existent movies")
            return False
        
        print("✅ Data consistency checks passed")
        return True
        
    except Exception as e:
        print(f"❌ Error during consistency check: {e}")
        return False

def display_sample_data():
    """Display sample data for manual verification."""
    print("\n📊 Sample Data Preview:")
    
    try:
        # Show sample movies
        with open('data/initial_movies.json', 'r', encoding='utf-8') as f:
            initial_movies = json.load(f)
        
        print("\n🎬 Sample Initial Movies:")
        for i, movie in enumerate(initial_movies[:5]):
            print(f"  {i+1}. {movie['CleanTitle']} ({movie['Year']}) - {movie['GenreList']}")
            print(f"     Ratings: {movie['rating_count']}, Avg: {movie['avg_rating']:.2f}")
        
        # Show genre distribution
        all_genres = {}
        for movie in initial_movies:
            for genre in movie['GenreList']:
                all_genres[genre] = all_genres.get(genre, 0) + 1
        
        print(f"\n🎭 Genre Distribution in Initial Movies:")
        for genre, count in sorted(all_genres.items(), key=lambda x: x[1], reverse=True):
            print(f"  {genre}: {count} movies")
        
        # Show ratings sample
        ratings = pd.read_csv('data/ratings.csv')
        print(f"\n⭐ Ratings Overview:")
        print(f"  Total ratings: {len(ratings):,}")
        print(f"  Unique users: {ratings['UserID'].nunique():,}")
        print(f"  Unique movies: {ratings['MovieID'].nunique():,}")
        print(f"  Rating distribution:")
        for rating in sorted(ratings['Rating'].unique()):
            count = len(ratings[ratings['Rating'] == rating])
            percentage = (count / len(ratings)) * 100
            print(f"    {rating} stars: {count:,} ({percentage:.1f}%)")
        
    except Exception as e:
        print(f"❌ Error displaying sample data: {e}")

def main():
    """Run all data verification tests."""
    print("🧪 MovieLens Data Processing Verification")
    print("=" * 50)
    
    tests = [
        test_file_existence,
        test_movies_json,
        test_initial_movies_json,
        test_ratings_csv,
        test_data_summary_json,
        test_data_consistency
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test in tests:
        if test():
            passed_tests += 1
        else:
            print()  # Add spacing after failed tests
    
    print("\n" + "=" * 50)
    print(f"📈 Test Results: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! Data processing was successful.")
        display_sample_data()
        return True
    else:
        print("❌ Some tests failed. Please review the data processing.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)