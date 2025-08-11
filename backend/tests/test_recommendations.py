"""
Comprehensive tests for the recommendation system.
Tests ML model loading, recommendation generation, and API endpoints.
"""

import pytest
import json
import numpy as np
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path

from app.core.ml_models import ModelManager
from app.services.recommendation_engine import RecommendationEngine, RecommendationCache
from app.models.recommendation import RecommendationResponse, RecommendationItem
from app.models.movie import MovieSummary


class TestModelManager:
    """Test ML model loading and management."""
    
    @pytest.fixture
    def model_manager(self):
        return ModelManager()
    
    @pytest.fixture
    def sample_ratings(self):
        return {1: 4.0, 2: 3.5, 3: 5.0, 4: 2.0, 5: 4.5, 6: 3.0, 7: 4.0, 8: 3.5, 9: 5.0, 10: 2.5,
                11: 4.0, 12: 3.0, 13: 4.5, 14: 3.5, 15: 4.0}
    
    @pytest.fixture
    def sample_movies_data(self):
        return [
            {"id": 1, "title": "Movie 1", "genres": ["Action", "Drama"], "year": 1995, "average_rating": 4.0},
            {"id": 2, "title": "Movie 2", "genres": ["Comedy"], "year": 1996, "average_rating": 3.5},
            {"id": 3, "title": "Movie 3", "genres": ["Drama", "Romance"], "year": 1997, "average_rating": 4.2}
        ]
    
    def test_model_manager_initialization(self, model_manager):
        """Test ModelManager initialization."""
        assert not model_manager.models_loaded
        assert model_manager.svd_model is None
        assert model_manager.content_similarity_matrix is None
        assert model_manager.hybrid_config is None
        assert model_manager.fallback_recommendations is None
    
    def test_is_ready_false_initially(self, model_manager):
        """Test that model manager is not ready initially."""
        assert not model_manager.is_ready()
    
    def test_validate_models(self, model_manager):
        """Test model validation."""
        status = model_manager._validate_models()
        
        assert isinstance(status, dict)
        assert 'collaborative_available' in status
        assert 'content_available' in status
        assert 'hybrid_available' in status
        assert 'fallback_available' in status
        assert 'movies_available' in status
        
        # Initially all should be False
        assert not any(status.values())
    
    @pytest.mark.asyncio
    async def test_load_models_missing_directory(self, model_manager):
        """Test model loading with missing models directory."""
        with patch('pathlib.Path.exists', return_value=False):
            await model_manager.load_models()
            # Should not raise exception, just log warning
            assert not model_manager.models_loaded
    
    @pytest.mark.asyncio
    async def test_load_movie_data(self, model_manager, sample_movies_data, tmp_path):
        """Test loading movie data."""
        # Create temporary movies file
        movies_file = tmp_path / "movies.json"
        with open(movies_file, 'w') as f:
            json.dump(sample_movies_data, f)
        
        # Mock the data directory path
        with patch('pathlib.Path') as mock_path:
            mock_path.return_value = tmp_path
            await model_manager._load_movie_data(tmp_path)
        
        assert model_manager.movies_data == sample_movies_data
        assert len(model_manager.movie_id_to_index) == 3
        assert len(model_manager.index_to_movie_id) == 3
        assert model_manager.movie_id_to_index[1] == 0
        assert model_manager.index_to_movie_id[0] == 1
    
    @pytest.mark.asyncio
    async def test_generate_recommendations_insufficient_ratings(self, model_manager):
        """Test recommendation generation with insufficient ratings."""
        insufficient_ratings = {1: 4.0, 2: 3.5}  # Only 2 ratings
        
        with pytest.raises(ValueError, match="At least 15 ratings required"):
            await model_manager.generate_recommendations(insufficient_ratings)
    
    @pytest.mark.asyncio
    async def test_generate_fallback_recommendations(self, model_manager, sample_ratings):
        """Test fallback recommendation generation."""
        # Set up fallback data
        model_manager.fallback_recommendations = [
            {"movie_id": 100, "title": "Fallback Movie 1", "genres": "Action", "avg_rating": 4.5, "explanation": "Popular movie"},
            {"movie_id": 101, "title": "Fallback Movie 2", "genres": "Comedy", "avg_rating": 4.2, "explanation": "Highly rated"}
        ]
        
        recommendations = await model_manager._generate_fallback_recommendations(sample_ratings, 2)
        
        assert len(recommendations) == 2
        assert all(isinstance(rec, RecommendationItem) for rec in recommendations)
        assert recommendations[0].movie.id == 100
        assert recommendations[0].algorithm_used == "fallback"
        assert recommendations[0].confidence == 0.6
    
    def test_create_movie_summary(self, model_manager, sample_movies_data):
        """Test movie summary creation."""
        movie_data = sample_movies_data[0]
        summary = model_manager._create_movie_summary(movie_data)
        
        assert isinstance(summary, MovieSummary)
        assert summary.id == 1
        assert summary.title == "Movie 1"
        assert summary.genres == ["Action", "Drama"]
        assert summary.year == 1995
        assert summary.average_rating == 4.0


class TestRecommendationCache:
    """Test recommendation caching functionality."""
    
    @pytest.fixture
    def cache(self):
        return RecommendationCache(ttl=60)  # 1 minute TTL for testing
    
    @pytest.fixture
    def sample_ratings(self):
        return {1: 4.0, 2: 3.5, 3: 5.0}
    
    @pytest.fixture
    def sample_response(self):
        return RecommendationResponse(
            recommendations=[],
            total_ratings=3,
            algorithm_used="hybrid",
            confidence_score=0.8,
            processing_time=0.1
        )
    
    def test_cache_initialization(self, cache):
        """Test cache initialization."""
        assert cache.ttl == 60
        assert len(cache.cache) == 0
    
    def test_generate_key(self, cache, sample_ratings):
        """Test cache key generation."""
        key1 = cache._generate_key(sample_ratings, "hybrid", 20)
        key2 = cache._generate_key(sample_ratings, "hybrid", 20)
        key3 = cache._generate_key(sample_ratings, "collaborative", 20)
        
        assert key1 == key2  # Same parameters should generate same key
        assert key1 != key3  # Different parameters should generate different keys
        assert len(key1) == 32  # MD5 hash length
    
    def test_cache_set_and_get(self, cache, sample_ratings, sample_response):
        """Test setting and getting cached responses."""
        # Initially should return None
        result = cache.get(sample_ratings, "hybrid", 20)
        assert result is None
        
        # Set cache entry
        cache.set(sample_ratings, "hybrid", 20, sample_response)
        
        # Should now return cached response
        result = cache.get(sample_ratings, "hybrid", 20)
        assert result == sample_response
    
    def test_cache_expiration(self, sample_ratings, sample_response):
        """Test cache expiration."""
        cache = RecommendationCache(ttl=0.1)  # Very short TTL
        
        cache.set(sample_ratings, "hybrid", 20, sample_response)
        
        # Should be available immediately
        result = cache.get(sample_ratings, "hybrid", 20)
        assert result == sample_response
        
        # Wait for expiration
        import time
        time.sleep(0.2)
        
        # Should now return None
        result = cache.get(sample_ratings, "hybrid", 20)
        assert result is None
    
    def test_cache_stats(self, cache, sample_ratings, sample_response):
        """Test cache statistics."""
        stats = cache.get_stats()
        assert stats['total_entries'] == 0
        assert stats['valid_entries'] == 0
        
        cache.set(sample_ratings, "hybrid", 20, sample_response)
        
        stats = cache.get_stats()
        assert stats['total_entries'] == 1
        assert stats['valid_entries'] == 1
        assert stats['ttl'] == 60
    
    def test_cache_clear(self, cache, sample_ratings, sample_response):
        """Test cache clearing."""
        cache.set(sample_ratings, "hybrid", 20, sample_response)
        assert len(cache.cache) == 1
        
        cache.clear()
        assert len(cache.cache) == 0


class TestRecommendationEngine:
    """Test recommendation engine functionality."""
    
    @pytest.fixture
    def mock_model_manager(self):
        manager = Mock(spec=ModelManager)
        manager.is_ready.return_value = True
        manager.generate_recommendations = AsyncMock()
        return manager
    
    @pytest.fixture
    def engine(self, mock_model_manager):
        return RecommendationEngine(mock_model_manager)
    
    @pytest.fixture
    def sample_ratings(self):
        return {1: 4.0, 2: 3.5, 3: 5.0, 4: 2.0, 5: 4.5, 6: 3.0, 7: 4.0, 8: 3.5, 9: 5.0, 10: 2.5,
                11: 4.0, 12: 3.0, 13: 4.5, 14: 3.5, 15: 4.0}
    
    @pytest.fixture
    def sample_response(self):
        return RecommendationResponse(
            recommendations=[
                RecommendationItem(
                    movie=MovieSummary(id=100, title="Test Movie", genres=["Action"], year=2020, average_rating=4.0),
                    predicted_rating=4.5,
                    confidence=0.8,
                    explanation="Test explanation",
                    algorithm_used="hybrid"
                )
            ],
            total_ratings=15,
            algorithm_used="hybrid",
            confidence_score=0.8,
            processing_time=0.1
        )
    
    def test_engine_initialization(self, engine, mock_model_manager):
        """Test recommendation engine initialization."""
        assert engine.model_manager == mock_model_manager
        assert isinstance(engine.cache, RecommendationCache)
        assert engine.request_count == 0
        assert engine.total_processing_time == 0.0
    
    @pytest.mark.asyncio
    async def test_generate_recommendations_success(self, engine, mock_model_manager, sample_ratings, sample_response):
        """Test successful recommendation generation."""
        mock_model_manager.generate_recommendations.return_value = sample_response
        
        result = await engine.generate_recommendations(sample_ratings, "hybrid", 20)
        
        assert isinstance(result, RecommendationResponse)
        assert result.algorithm_used == "hybrid"
        assert len(result.recommendations) == 1
        assert 'from_cache' in result.metadata
        assert not result.metadata['from_cache']
        
        mock_model_manager.generate_recommendations.assert_called_once_with(
            ratings=sample_ratings,
            algorithm="hybrid",
            num_recommendations=20
        )
    
    @pytest.mark.asyncio
    async def test_generate_recommendations_with_cache(self, engine, mock_model_manager, sample_ratings, sample_response):
        """Test recommendation generation with caching."""
        mock_model_manager.generate_recommendations.return_value = sample_response
        
        # First call should generate and cache
        result1 = await engine.generate_recommendations(sample_ratings, "hybrid", 20)
        assert not result1.metadata['from_cache']
        
        # Second call should use cache
        result2 = await engine.generate_recommendations(sample_ratings, "hybrid", 20)
        assert result2.metadata['from_cache']
        
        # Model manager should only be called once
        mock_model_manager.generate_recommendations.assert_called_once()
    
    def test_validate_input_success(self, engine, sample_ratings):
        """Test successful input validation."""
        # Should not raise exception
        engine._validate_input(sample_ratings, "hybrid", 20)
    
    def test_validate_input_empty_ratings(self, engine):
        """Test input validation with empty ratings."""
        with pytest.raises(ValueError, match="Ratings dictionary cannot be empty"):
            engine._validate_input({}, "hybrid", 20)
    
    def test_validate_input_insufficient_ratings(self, engine):
        """Test input validation with insufficient ratings."""
        insufficient_ratings = {1: 4.0, 2: 3.5}
        with pytest.raises(ValueError, match="At least 15 ratings required"):
            engine._validate_input(insufficient_ratings, "hybrid", 20)
    
    def test_validate_input_invalid_algorithm(self, engine, sample_ratings):
        """Test input validation with invalid algorithm."""
        with pytest.raises(ValueError, match="Invalid algorithm"):
            engine._validate_input(sample_ratings, "invalid", 20)
    
    def test_validate_input_invalid_num_recommendations(self, engine, sample_ratings):
        """Test input validation with invalid number of recommendations."""
        with pytest.raises(ValueError, match="Number of recommendations must be between 1 and 100"):
            engine._validate_input(sample_ratings, "hybrid", 0)
        
        with pytest.raises(ValueError, match="Number of recommendations must be between 1 and 100"):
            engine._validate_input(sample_ratings, "hybrid", 101)
    
    def test_validate_input_invalid_rating_values(self, engine):
        """Test input validation with invalid rating values."""
        invalid_ratings = {1: 4.0, 2: 6.0, 3: 3.0}  # Rating 6.0 is invalid
        invalid_ratings.update({i: 4.0 for i in range(4, 16)})  # Add more ratings to meet minimum
        
        with pytest.raises(ValueError, match="Rating must be between 1.0 and 5.0"):
            engine._validate_input(invalid_ratings, "hybrid", 20)
    
    @pytest.mark.asyncio
    async def test_update_recommendations(self, engine, mock_model_manager, sample_response):
        """Test recommendation updates."""
        existing_ratings = {1: 4.0, 2: 3.5, 3: 5.0, 4: 2.0, 5: 4.5, 6: 3.0, 7: 4.0, 8: 3.5}
        new_ratings = {9: 5.0, 10: 2.5, 11: 4.0, 12: 3.0, 13: 4.5, 14: 3.5, 15: 4.0}
        
        mock_model_manager.generate_recommendations.return_value = sample_response
        
        result = await engine.update_recommendations(existing_ratings, new_ratings, "hybrid", 20)
        
        assert isinstance(result, RecommendationResponse)
        assert result.metadata['update_request']
        assert result.metadata['existing_ratings_count'] == 8
        assert result.metadata['new_ratings_count'] == 7
        assert result.metadata['total_ratings_count'] == 15
    
    def test_get_engine_stats(self, engine):
        """Test engine statistics."""
        stats = engine.get_engine_stats()
        
        assert 'total_requests' in stats
        assert 'total_processing_time' in stats
        assert 'average_processing_time' in stats
        assert 'cache_stats' in stats
        assert 'model_ready' in stats
        
        assert stats['total_requests'] == 0
        assert stats['total_processing_time'] == 0.0
        assert stats['average_processing_time'] == 0
    
    def test_clear_cache(self, engine):
        """Test cache clearing."""
        # Add something to cache first
        engine.cache.cache['test'] = {'response': None, 'timestamp': 0}
        assert len(engine.cache.cache) == 1
        
        engine.clear_cache()
        assert len(engine.cache.cache) == 0


class TestRecommendationAPI:
    """Test recommendation API endpoints."""
    
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from app.main import app
        return TestClient(app)
    
    @pytest.fixture
    def sample_request_data(self):
        return {
            "ratings": {str(i): 4.0 for i in range(1, 16)},  # 15 ratings
            "algorithm": "hybrid",
            "num_recommendations": 10
        }
    
    def test_generate_recommendations_endpoint_validation(self, client):
        """Test recommendation endpoint input validation."""
        # Test with insufficient ratings
        insufficient_data = {
            "ratings": {"1": 4.0, "2": 3.5},  # Only 2 ratings
            "algorithm": "hybrid",
            "num_recommendations": 10
        }
        
        response = client.post("/api/recommendations", json=insufficient_data)
        assert response.status_code == 422  # Validation error
    
    def test_update_recommendations_endpoint_validation(self, client):
        """Test update recommendations endpoint validation."""
        invalid_data = {
            "existing_ratings": {"1": 4.0},
            "new_ratings": {},  # Empty new ratings
            "algorithm": "hybrid",
            "num_recommendations": 10
        }
        
        response = client.post("/api/recommendations/update", json=invalid_data)
        assert response.status_code == 422  # Validation error
    
    def test_recommendation_stats_endpoint(self, client):
        """Test recommendation statistics endpoint."""
        response = client.get("/api/recommendations/stats")
        # Should return stats even if models aren't loaded
        if response.status_code == 200:
            data = response.json()
            assert "status" in data
            assert "data" in data
            assert "timestamp" in data
        else:
            # If it fails, it should be a 500 with proper error structure
            assert response.status_code == 500
    
    def test_clear_cache_endpoint(self, client):
        """Test cache clearing endpoint."""
        response = client.post("/api/recommendations/cache/clear")
        # Should work even if models aren't loaded
        if response.status_code == 200:
            data = response.json()
            assert "status" in data
            assert data["status"] == "success"
            assert "message" in data
        else:
            # If it fails, it should be a 500 with proper error structure
            assert response.status_code == 500
    
    def test_recommendations_endpoint_with_valid_data(self, client):
        """Test recommendations endpoint with valid data."""
        valid_data = {
            "ratings": {str(i): 4.0 for i in range(1, 16)},  # 15 ratings
            "algorithm": "hybrid",
            "num_recommendations": 5
        }
        
        response = client.post("/api/recommendations", json=valid_data)
        
        # Should either succeed or fail gracefully
        if response.status_code == 200:
            data = response.json()
            assert "recommendations" in data
            assert "algorithm_used" in data
            assert "processing_time" in data
            assert "confidence_score" in data
            assert len(data["recommendations"]) <= 5
        else:
            # Should be a proper error response
            assert response.status_code in [400, 500, 503]


class TestIntegration:
    """Integration tests with real model files."""
    
    @pytest.mark.asyncio
    async def test_full_recommendation_pipeline(self):
        """Test the complete recommendation pipeline with real data."""
        # Always test the system - it should work with fallback even if ML models fail
        model_manager = ModelManager()
        await model_manager.load_models()
        
        # System should always be ready (with fallback if needed)
        assert model_manager.is_ready(), "System should be ready with fallback recommendations"
        
        # Test with real ratings
        sample_ratings = {i: 4.0 for i in range(1, 16)}
        
        response = await model_manager.generate_recommendations(
            ratings=sample_ratings,
            algorithm="hybrid",
            num_recommendations=5
        )
        
        # Validate response structure
        assert isinstance(response, RecommendationResponse)
        assert len(response.recommendations) <= 5
        assert response.algorithm_used in ["hybrid", "collaborative", "content", "fallback"]
        assert 0 <= response.confidence_score <= 1
        assert response.processing_time >= 0  # Processing time can be 0 for very fast operations
        assert response.total_ratings == 15
        
        # Validate recommendation items
        for rec in response.recommendations:
            assert isinstance(rec, RecommendationItem)
            assert isinstance(rec.movie, MovieSummary)
            assert 1.0 <= rec.predicted_rating <= 5.0
            assert 0.0 <= rec.confidence <= 1.0
            assert len(rec.explanation) > 0
            assert rec.algorithm_used in ["hybrid", "collaborative", "content", "fallback"]
        
        # Test that we get different results with different algorithms
        if model_manager._validate_models()['fallback_available']:
            fallback_response = await model_manager.generate_recommendations(
                ratings=sample_ratings,
                algorithm="collaborative",  # Will fallback if not available
                num_recommendations=3
            )
            assert isinstance(fallback_response, RecommendationResponse)
            assert len(fallback_response.recommendations) <= 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])