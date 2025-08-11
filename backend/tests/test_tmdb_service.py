"""
Tests for the TMDB service functionality.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import json
from pathlib import Path

from app.services.tmdb_service import TMDBService


class TestTMDBService:
    """Test cases for the TMDBService class."""
    
    @pytest.fixture
    def tmdb_service(self):
        """Create a test TMDB service instance."""
        with patch('app.core.config.settings') as mock_settings:
            mock_settings.TMDB_API_KEY = "test_api_key"
            mock_settings.TMDB_BASE_URL = "https://api.themoviedb.org/3"
            mock_settings.TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"
            
            service = TMDBService()
            service.cache_file = Path("test_tmdb_cache.json")  # Use test cache file
            return service
    
    @pytest.fixture
    def sample_tmdb_response(self):
        """Sample TMDB API response for testing."""
        return {
            "results": [
                {
                    "id": 862,
                    "title": "Toy Story",
                    "release_date": "1995-10-30",
                    "poster_path": "/uXDfjJbdP4ijW5hWSBrPrlKpxab.jpg",
                    "overview": "A cowboy doll is profoundly threatened..."
                }
            ]
        }
    
    def test_cache_key_generation(self, tmdb_service):
        """Test cache key generation."""
        key = tmdb_service._get_cache_key("Toy Story", 1995)
        assert key == "toy story_1995"
        
        key = tmdb_service._get_cache_key("  The Matrix  ", 1999)
        assert key == "the matrix_1999"
    
    def test_title_cleaning(self, tmdb_service):
        """Test title cleaning for search."""
        cleaned = tmdb_service._clean_title_for_search("The Matrix")
        assert cleaned == "Matrix"
        
        cleaned = tmdb_service._clean_title_for_search("Star Wars: Episode IV")
        assert cleaned == "Star Wars Episode IV"
        
        cleaned = tmdb_service._clean_title_for_search("Fast & Furious")
        assert cleaned == "Fast and Furious"
    
    def test_best_match_finding(self, tmdb_service, sample_tmdb_response):
        """Test finding best match from TMDB results."""
        results = sample_tmdb_response["results"]
        
        # Test exact year match
        best_match = tmdb_service._find_best_match(results, "Toy Story", 1995)
        assert best_match is not None
        assert best_match["title"] == "Toy Story"
        
        # Test no results
        best_match = tmdb_service._find_best_match([], "Nonexistent Movie", 2000)
        assert best_match is None
    
    def test_placeholder_poster_url(self, tmdb_service):
        """Test placeholder poster URL generation."""
        placeholder_url = tmdb_service.get_placeholder_poster_url()
        assert "placeholder" in placeholder_url.lower()
        assert "500x750" in placeholder_url
    
    @pytest.mark.asyncio
    async def test_successful_poster_fetch(self, tmdb_service, sample_tmdb_response):
        """Test successful poster fetching from TMDB."""
        # Ensure API key is set for this test
        tmdb_service.api_key = "test_api_key"
        
        with patch('httpx.AsyncClient') as mock_client_class:
            # Mock the async context manager and get method
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            mock_client_class.return_value.__aexit__.return_value = None
            
            # Mock HTTP response
            mock_response = Mock()
            mock_response.json.return_value = sample_tmdb_response
            mock_response.raise_for_status.return_value = None
            mock_client.get.return_value = mock_response
            
            poster_url = await tmdb_service.get_movie_poster_url("Toy Story", 1995)
            
            assert poster_url == "https://image.tmdb.org/t/p/w500/uXDfjJbdP4ijW5hWSBrPrlKpxab.jpg"
            
            # Check that result is cached
            cached_url = await tmdb_service.get_movie_poster_url("Toy Story", 1995)
            assert cached_url == poster_url
            
            # Should only call API once due to caching
            assert mock_client.get.call_count == 1
    
    @pytest.mark.asyncio
    async def test_no_results_poster_fetch(self, tmdb_service):
        """Test poster fetching when no results found."""
        # Ensure API key is set for this test
        tmdb_service.api_key = "test_api_key"
        
        with patch('httpx.AsyncClient') as mock_client_class:
            # Mock the async context manager and get method
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            mock_client_class.return_value.__aexit__.return_value = None
            
            # Mock HTTP response with no results
            mock_response = Mock()
            mock_response.json.return_value = {"results": []}
            mock_response.raise_for_status.return_value = None
            mock_client.get.return_value = mock_response
            
            poster_url = await tmdb_service.get_movie_poster_url("Nonexistent Movie", 2000)
            
            assert poster_url is None
            
            # Check that None result is cached
            cached_url = await tmdb_service.get_movie_poster_url("Nonexistent Movie", 2000)
            assert cached_url is None
    
    @pytest.mark.asyncio
    async def test_no_api_key(self, tmdb_service):
        """Test behavior when no API key is configured."""
        tmdb_service.api_key = None
        
        poster_url = await tmdb_service.get_movie_poster_url("Toy Story", 1995)
        
        assert poster_url is None
    
    @pytest.mark.asyncio
    async def test_api_error_handling(self, tmdb_service):
        """Test handling of API errors."""
        with patch('httpx.AsyncClient') as mock_client_class:
            # Mock the async context manager to raise an error
            mock_client_class.return_value.__aenter__.side_effect = Exception("API Error")
            
            poster_url = await tmdb_service.get_movie_poster_url("Toy Story", 1995)
            
            assert poster_url is None
            
            # Check that error result is cached
            cached_url = await tmdb_service.get_movie_poster_url("Toy Story", 1995)
            assert cached_url is None
    
    @pytest.mark.asyncio
    async def test_multiple_posters_batch(self, tmdb_service, sample_tmdb_response):
        """Test fetching multiple posters in batch."""
        # Ensure API key is set for this test
        tmdb_service.api_key = "test_api_key"
        
        with patch('httpx.AsyncClient') as mock_client_class:
            # Mock the async context manager and get method
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            mock_client_class.return_value.__aexit__.return_value = None
            
            # Mock HTTP response
            mock_response = Mock()
            mock_response.json.return_value = sample_tmdb_response
            mock_response.raise_for_status.return_value = None
            mock_client.get.return_value = mock_response
            
            movies = [
                {"id": 1, "title": "Toy Story", "year": 1995},
                {"id": 2, "title": "Jumanji", "year": 1995}
            ]
            
            poster_urls = await tmdb_service.get_multiple_posters(movies)
            
            assert isinstance(poster_urls, dict)
            assert len(poster_urls) == 2
            assert 1 in poster_urls
            assert 2 in poster_urls
    
    def test_cache_loading_and_saving(self, tmdb_service):
        """Test cache loading and saving functionality."""
        # Test saving cache
        tmdb_service.poster_cache = {"test_movie_2000": "http://test.com/poster.jpg"}
        tmdb_service._save_cache()
        
        # Test loading cache
        new_service = TMDBService()
        new_service.cache_file = tmdb_service.cache_file
        new_service._load_cache()
        
        assert "test_movie_2000" in new_service.poster_cache
        assert new_service.poster_cache["test_movie_2000"] == "http://test.com/poster.jpg"
        
        # Clean up test cache file
        if tmdb_service.cache_file.exists():
            tmdb_service.cache_file.unlink()
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self, tmdb_service, sample_tmdb_response):
        """Test that rate limiting is applied."""
        # Ensure API key is set for this test
        tmdb_service.api_key = "test_api_key"
        
        with patch('asyncio.sleep') as mock_sleep, \
             patch('httpx.AsyncClient') as mock_client_class:
            
            # Mock the async context manager and get method
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            mock_client_class.return_value.__aexit__.return_value = None
            
            # Mock HTTP response
            mock_response = Mock()
            mock_response.json.return_value = sample_tmdb_response
            mock_response.raise_for_status.return_value = None
            mock_client.get.return_value = mock_response
            
            await tmdb_service.get_movie_poster_url("Toy Story", 1995)
            
            # Should have called sleep for rate limiting
            mock_sleep.assert_called_with(0.25)
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, tmdb_service):
        """Test handling of concurrent requests to the same movie."""
        # Create multiple concurrent requests for the same movie
        tasks = [
            tmdb_service.get_movie_poster_url("Toy Story", 1995)
            for _ in range(5)
        ]
        
        results = await asyncio.gather(*tasks)
        
        # All results should be the same (either all None or all the same URL)
        assert all(result == results[0] for result in results)