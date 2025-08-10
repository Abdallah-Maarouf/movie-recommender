"""
Tests for Pydantic models.
"""

import pytest
from pydantic import ValidationError

from app.models.movie import Movie, MovieSummary, InitialMoviesResponse
from app.models.recommendation import (
    RatingRequest, 
    RecommendationItem, 
    RecommendationResponse,
    UpdateRecommendationRequest,
    ErrorResponse
)


class TestMovieModels:
    """Test cases for movie-related models."""
    
    def test_movie_summary_valid(self):
        """Test MovieSummary with valid data."""
        data = {
            "id": 1,
            "title": "Test Movie",
            "genres": ["Action", "Drama"],
            "year": 2020,
            "average_rating": 4.2
        }
        
        movie = MovieSummary(**data)
        
        assert movie.id == 1
        assert movie.title == "Test Movie"
        assert movie.genres == ["Action", "Drama"]
        assert movie.year == 2020
        assert movie.average_rating == 4.2
        assert movie.poster_url is None
    
    def test_movie_full_valid(self, sample_movie_data):
        """Test Movie with complete valid data."""
        movie = Movie(**sample_movie_data)
        
        assert movie.id == sample_movie_data["id"]
        assert movie.title == sample_movie_data["title"]
        assert movie.genres == sample_movie_data["genres"]
        assert movie.year == sample_movie_data["year"]
        assert movie.average_rating == sample_movie_data["average_rating"]
        assert movie.rating_count == sample_movie_data["rating_count"]
    
    def test_movie_invalid_year(self):
        """Test Movie with invalid year."""
        data = {
            "id": 1,
            "title": "Test Movie",
            "genres": ["Action"],
            "year": 1800,  # Too old
            "average_rating": 4.0,
            "rating_count": 100
        }
        
        with pytest.raises(ValidationError):
            Movie(**data)
    
    def test_movie_invalid_rating(self):
        """Test Movie with invalid rating."""
        data = {
            "id": 1,
            "title": "Test Movie",
            "genres": ["Action"],
            "year": 2020,
            "average_rating": 6.0,  # Too high
            "rating_count": 100
        }
        
        with pytest.raises(ValidationError):
            Movie(**data)
    
    def test_movie_empty_genres(self):
        """Test Movie with empty genres list."""
        data = {
            "id": 1,
            "title": "Test Movie",
            "genres": [],  # Empty genres
            "year": 2020,
            "average_rating": 4.0,
            "rating_count": 100
        }
        
        with pytest.raises(ValidationError):
            Movie(**data)


class TestRecommendationModels:
    """Test cases for recommendation-related models."""
    
    def test_rating_request_valid(self, sample_ratings):
        """Test RatingRequest with valid data."""
        request = RatingRequest(
            ratings=sample_ratings,
            algorithm="hybrid",
            num_recommendations=20
        )
        
        assert len(request.ratings) == len(sample_ratings)
        assert request.algorithm == "hybrid"
        assert request.num_recommendations == 20
    
    def test_rating_request_insufficient_ratings(self):
        """Test RatingRequest with insufficient ratings."""
        ratings = {1: 4.0, 2: 3.0}  # Only 2 ratings
        
        with pytest.raises(ValidationError):
            RatingRequest(ratings=ratings)
    
    def test_rating_request_invalid_rating_value(self):
        """Test RatingRequest with invalid rating values."""
        ratings = {i: 4.0 for i in range(1, 16)}  # 15 valid ratings
        ratings[16] = 6.0  # Invalid rating value
        
        with pytest.raises(ValidationError):
            RatingRequest(ratings=ratings)
    
    def test_rating_request_invalid_movie_id(self):
        """Test RatingRequest with invalid movie ID."""
        ratings = {i: 4.0 for i in range(1, 16)}  # 15 valid ratings
        ratings[-1] = 4.0  # Invalid movie ID (negative)
        
        with pytest.raises(ValidationError):
            RatingRequest(ratings=ratings)
    
    def test_recommendation_item_valid(self):
        """Test RecommendationItem with valid data."""
        movie_data = {
            "id": 1,
            "title": "Test Movie",
            "genres": ["Action"],
            "year": 2020,
            "average_rating": 4.0
        }
        
        item = RecommendationItem(
            movie=MovieSummary(**movie_data),
            predicted_rating=4.2,
            confidence=0.85,
            explanation="Because you liked similar movies",
            algorithm_used="hybrid"
        )
        
        assert item.predicted_rating == 4.2
        assert item.confidence == 0.85
        assert item.explanation == "Because you liked similar movies"
        assert item.algorithm_used == "hybrid"
    
    def test_recommendation_response_valid(self):
        """Test RecommendationResponse with valid data."""
        movie_data = {
            "id": 1,
            "title": "Test Movie",
            "genres": ["Action"],
            "year": 2020,
            "average_rating": 4.0
        }
        
        recommendation = RecommendationItem(
            movie=MovieSummary(**movie_data),
            predicted_rating=4.2,
            confidence=0.85,
            explanation="Test explanation",
            algorithm_used="hybrid"
        )
        
        response = RecommendationResponse(
            recommendations=[recommendation],
            total_ratings=18,
            algorithm_used="hybrid",
            confidence_score=0.82,
            processing_time=0.245
        )
        
        assert len(response.recommendations) == 1
        assert response.total_ratings == 18
        assert response.algorithm_used == "hybrid"
        assert response.confidence_score == 0.82
        assert response.processing_time == 0.245
    
    def test_update_recommendation_request_valid(self):
        """Test UpdateRecommendationRequest with valid data."""
        existing_ratings = {i: 4.0 for i in range(1, 16)}
        new_ratings = {16: 3.0, 17: 5.0}
        
        request = UpdateRecommendationRequest(
            existing_ratings=existing_ratings,
            new_ratings=new_ratings,
            algorithm="hybrid"
        )
        
        assert len(request.existing_ratings) == 15
        assert len(request.new_ratings) == 2
        assert request.algorithm == "hybrid"
    
    def test_error_response_valid(self):
        """Test ErrorResponse with valid data."""
        error = ErrorResponse(
            error="TEST_ERROR",
            message="This is a test error",
            status_code=400,
            timestamp=1642234567.123
        )
        
        assert error.error == "TEST_ERROR"
        assert error.message == "This is a test error"
        assert error.status_code == 400
        assert error.timestamp == 1642234567.123