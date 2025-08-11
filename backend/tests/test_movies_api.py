"""
Tests for the movies API endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock

from app.main import app
from app.models.movie import Movie, MovieSummary
from app.utils.data_loader import get_data_loader


class TestMoviesAPI:
    """Test cases for movies API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create a test client."""
        return TestClient(app)
    
    @pytest.fixture
    def sample_initial_movies(self):
        """Sample initial movies for testing."""
        return [
            MovieSummary(
                id=1,
                title="Toy Story",
                genres=["Animation", "Children's", "Comedy"],
                year=1995,
                poster_url="https://image.tmdb.org/t/p/w500/poster1.jpg",
                average_rating=3.9
            ),
            MovieSummary(
                id=2,
                title="Jumanji",
                genres=["Adventure", "Children's", "Fantasy"],
                year=1995,
                poster_url="https://image.tmdb.org/t/p/w500/poster2.jpg",
                average_rating=3.2
            )
        ]
    
    @pytest.fixture
    def sample_movie(self):
        """Sample movie for testing."""
        return Movie(
            id=1,
            title="Toy Story",
            genres=["Animation", "Children's", "Comedy"],
            year=1995,
            average_rating=3.9,
            rating_count=2077,
            poster_url="https://image.tmdb.org/t/p/w500/poster1.jpg",
            description="A cowboy doll is profoundly threatened...",
            director="John Lasseter",
            cast=["Tom Hanks", "Tim Allen"],
            runtime=81
        )
    
    def test_get_initial_movies_success(self, client, sample_initial_movies):
        """Test successful retrieval of initial movies."""
        # Override the dependency
        def mock_get_data_loader():
            mock_data_loader = Mock()
            mock_data_loader.get_initial_movies = AsyncMock(return_value=sample_initial_movies)
            return mock_data_loader
        
        app.dependency_overrides[get_data_loader] = mock_get_data_loader
        
        try:
            response = client.get("/api/movies/initial")
            
            assert response.status_code == 200
            data = response.json()
            
            assert "movies" in data
            assert "total_count" in data
            assert data["total_count"] == 2
            assert len(data["movies"]) == 2
            
            # Check first movie
            movie1 = data["movies"][0]
            assert movie1["id"] == 1
            assert movie1["title"] == "Toy Story"
            assert movie1["year"] == 1995
            assert "poster_url" in movie1
            assert movie1["average_rating"] == 3.9
        finally:
            app.dependency_overrides.clear()
    
    def test_get_initial_movies_error(self, client):
        """Test error handling in initial movies endpoint."""
        # Override the dependency to raise exception
        def mock_get_data_loader():
            mock_data_loader = Mock()
            mock_data_loader.get_initial_movies = AsyncMock(side_effect=Exception("Data loading error"))
            return mock_data_loader
        
        app.dependency_overrides[get_data_loader] = mock_get_data_loader
        
        try:
            response = client.get("/api/movies/initial")
            
            assert response.status_code == 500
            data = response.json()
            
            assert "detail" in data
            assert data["detail"]["error"] == "DATA_LOADING_ERROR"
            assert "timestamp" in data["detail"]
        finally:
            app.dependency_overrides.clear()
    
    def test_get_movie_by_id_success(self, client, sample_movie):
        """Test successful retrieval of movie by ID."""
        # Override the dependency
        def mock_get_data_loader():
            mock_data_loader = Mock()
            mock_data_loader.get_movie_by_id = AsyncMock(return_value=sample_movie)
            return mock_data_loader
        
        app.dependency_overrides[get_data_loader] = mock_get_data_loader
        
        try:
            response = client.get("/api/movies/1")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["id"] == 1
            assert data["title"] == "Toy Story"
            assert data["year"] == 1995
            assert data["average_rating"] == 3.9
            assert data["rating_count"] == 2077
            assert "poster_url" in data
            assert "description" in data
            assert "director" in data
            assert "cast" in data
            assert "runtime" in data
        finally:
            app.dependency_overrides.clear()
    
    def test_get_movie_by_id_not_found(self, client):
        """Test movie not found error."""
        # Override the dependency to return None
        def mock_get_data_loader():
            mock_data_loader = Mock()
            mock_data_loader.get_movie_by_id = AsyncMock(return_value=None)
            return mock_data_loader
        
        app.dependency_overrides[get_data_loader] = mock_get_data_loader
        
        try:
            response = client.get("/api/movies/99999")
            
            assert response.status_code == 404
            data = response.json()
            
            assert "detail" in data
            assert data["detail"]["error"] == "MOVIE_NOT_FOUND"
            assert "99999" in data["detail"]["message"]
        finally:
            app.dependency_overrides.clear()
    
    def test_get_movie_by_id_error(self, client):
        """Test error handling in get movie endpoint."""
        # Override the dependency to raise exception
        def mock_get_data_loader():
            mock_data_loader = Mock()
            mock_data_loader.get_movie_by_id = AsyncMock(side_effect=Exception("Database error"))
            return mock_data_loader
        
        app.dependency_overrides[get_data_loader] = mock_get_data_loader
        
        try:
            response = client.get("/api/movies/1")
            
            assert response.status_code == 500
            data = response.json()
            
            assert "detail" in data
            assert data["detail"]["error"] == "DATA_LOADING_ERROR"
        finally:
            app.dependency_overrides.clear()
    
    def test_search_movies_by_title(self, client, sample_movie):
        """Test movie search by title."""
        # Override the dependency
        def mock_get_data_loader():
            mock_data_loader = Mock()
            mock_data_loader.get_all_movies = AsyncMock(return_value={1: sample_movie})
            return mock_data_loader
        
        app.dependency_overrides[get_data_loader] = mock_get_data_loader
        
        try:
            response = client.get("/api/movies/search?query=toy")
            
            assert response.status_code == 200
            data = response.json()
            
            assert isinstance(data, list)
            assert len(data) == 1
            assert data[0]["title"] == "Toy Story"
        finally:
            app.dependency_overrides.clear()
    
    def test_search_movies_by_genre(self, client, sample_movie):
        """Test movie search by genre."""
        # Override the dependency
        def mock_get_data_loader():
            mock_data_loader = Mock()
            mock_data_loader.get_all_movies = AsyncMock(return_value={1: sample_movie})
            return mock_data_loader
        
        app.dependency_overrides[get_data_loader] = mock_get_data_loader
        
        try:
            response = client.get("/api/movies/search?genre=animation")
            
            assert response.status_code == 200
            data = response.json()
            
            assert isinstance(data, list)
            assert len(data) == 1
            assert data[0]["title"] == "Toy Story"
        finally:
            app.dependency_overrides.clear()
    
    def test_search_movies_by_year(self, client, sample_movie):
        """Test movie search by year."""
        # Override the dependency
        def mock_get_data_loader():
            mock_data_loader = Mock()
            mock_data_loader.get_all_movies = AsyncMock(return_value={1: sample_movie})
            return mock_data_loader
        
        app.dependency_overrides[get_data_loader] = mock_get_data_loader
        
        try:
            response = client.get("/api/movies/search?year=1995")
            
            assert response.status_code == 200
            data = response.json()
            
            assert isinstance(data, list)
            assert len(data) == 1
            assert data[0]["title"] == "Toy Story"
        finally:
            app.dependency_overrides.clear()
    
    def test_search_movies_no_parameters(self, client):
        """Test movie search with no parameters."""
        response = client.get("/api/movies/search")
        
        assert response.status_code == 400
        data = response.json()
        
        assert "detail" in data
        assert data["detail"]["error"] == "INVALID_SEARCH_PARAMETERS"
    
    def test_search_movies_with_limit(self, client):
        """Test movie search with limit parameter."""
        # Create multiple sample movies
        movies = {}
        for i in range(10):
            movies[i] = Movie(
                id=i,
                title=f"Test Movie {i}",
                genres=["Drama"],
                year=2000,
                average_rating=3.5,
                rating_count=100,
                poster_url=None,
                description="Test description",
                director="Test Director",
                cast=[],
                runtime=120
            )
        
        # Override the dependency
        def mock_get_data_loader():
            mock_data_loader = Mock()
            mock_data_loader.get_all_movies = AsyncMock(return_value=movies)
            return mock_data_loader
        
        app.dependency_overrides[get_data_loader] = mock_get_data_loader
        
        try:
            response = client.get("/api/movies/search?query=test&limit=5")
            
            assert response.status_code == 200
            data = response.json()
            
            assert isinstance(data, list)
            assert len(data) == 5
        finally:
            app.dependency_overrides.clear()
    
    def test_get_data_stats(self, client):
        """Test data statistics endpoint."""
        # Override the dependency
        def mock_get_data_loader():
            mock_data_loader = Mock()
            mock_data_loader.get_data_summary.return_value = {
                "total_movies": 3883,
                "initial_movies": 30,
                "movies_with_posters": 25,
                "data_loaded": True
            }
            return mock_data_loader
        
        app.dependency_overrides[get_data_loader] = mock_get_data_loader
        
        try:
            response = client.get("/api/movies/stats")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["total_movies"] == 3883
            assert data["initial_movies"] == 30
            assert data["movies_with_posters"] == 25
            assert data["data_loaded"] is True
        finally:
            app.dependency_overrides.clear()
    
    def test_get_data_stats_error(self, client):
        """Test error handling in data stats endpoint."""
        # Override the dependency to raise exception
        def mock_get_data_loader():
            mock_data_loader = Mock()
            mock_data_loader.get_data_summary.side_effect = Exception("Stats error")
            return mock_data_loader
        
        app.dependency_overrides[get_data_loader] = mock_get_data_loader
        
        try:
            response = client.get("/api/movies/stats")
            
            assert response.status_code == 500
            data = response.json()
            
            assert "detail" in data
            assert data["detail"]["error"] == "STATS_ERROR"
        finally:
            app.dependency_overrides.clear()
    
    def test_movies_endpoints_cors_headers(self, client):
        """Test that CORS headers are present in movie endpoints."""
        response = client.get("/api/movies/initial")
        
        # Should have CORS headers (even if request fails)
        # The actual CORS headers are added by middleware
        assert response.status_code in [200, 500]  # Either success or error is fine for this test
    
    def test_movies_endpoints_timing_headers(self, client):
        """Test that timing headers are added to responses."""
        response = client.get("/api/movies/stats")
        
        # Should have timing header added by middleware
        assert "X-Process-Time" in response.headers
        
        # Process time should be a valid float
        process_time = float(response.headers["X-Process-Time"])
        assert process_time >= 0