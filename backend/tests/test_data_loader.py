"""
Tests for the data loader functionality.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import json
from pathlib import Path

from app.utils.data_loader import DataLoader
from app.models.movie import Movie, MovieSummary


class TestDataLoader:
    """Test cases for the DataLoader class."""
    
    @pytest.fixture
    def data_loader(self):
        """Create a test data loader instance."""
        loader = DataLoader()
        return loader
    
    @pytest.fixture
    def sample_movies_data(self):
        """Sample movies data for testing."""
        return [
            {
                "MovieID": 1,
                "Title": "Toy Story (1995)",
                "Genres": "Animation|Children's|Comedy",
                "Year": 1995,
                "CleanTitle": "Toy Story",
                "GenreList": ["Animation", "Children's", "Comedy"],
                "WebTitle": "Toy Story"
            },
            {
                "MovieID": 2,
                "Title": "Jumanji (1995)",
                "Genres": "Adventure|Children's|Fantasy",
                "Year": 1995,
                "CleanTitle": "Jumanji",
                "GenreList": ["Adventure", "Children's", "Fantasy"],
                "WebTitle": "Jumanji"
            }
        ]
    
    @pytest.fixture
    def sample_initial_movies_data(self):
        """Sample initial movies data for testing."""
        return [
            {
                "MovieID": 1,
                "CleanTitle": "Toy Story",
                "WebTitle": "Toy Story",
                "Year": 1995,
                "GenreList": ["Animation", "Children's", "Comedy"],
                "rating_count": 2077.0,
                "avg_rating": 3.9
            },
            {
                "MovieID": 2,
                "CleanTitle": "Jumanji",
                "WebTitle": "Jumanji",
                "Year": 1995,
                "GenreList": ["Adventure", "Children's", "Fantasy"],
                "rating_count": 1500.0,
                "avg_rating": 3.2
            }
        ]
    
    @pytest.mark.asyncio
    async def test_placeholder_data_creation(self, data_loader):
        """Test that placeholder data is created when files don't exist."""
        # Mock the data directory to not exist
        with patch('pathlib.Path.exists', return_value=False):
            await data_loader.load_data()
        
        assert data_loader.data_loaded is True
        assert data_loader.initial_movies_cache is not None
        assert len(data_loader.initial_movies_cache) == 30
        assert data_loader.movies_cache is not None
        assert len(data_loader.movies_cache) == 30
    
    @pytest.mark.asyncio
    @patch('pathlib.Path.exists')
    @patch('app.services.tmdb_service.TMDBService.get_multiple_posters')
    async def test_real_data_loading(self, mock_get_posters, mock_exists, 
                                   data_loader, sample_movies_data, sample_initial_movies_data):
        """Test loading real data from JSON files."""
        # Mock file existence
        mock_exists.return_value = True
        
        # Mock file reading with proper mock_open
        from unittest.mock import mock_open
        
        def side_effect(file_path, *args, **kwargs):
            if 'movies.json' in str(file_path):
                return mock_open(read_data=json.dumps(sample_movies_data)).return_value
            elif 'initial_movies.json' in str(file_path):
                return mock_open(read_data=json.dumps(sample_initial_movies_data)).return_value
            return mock_open().return_value
        
        with patch('builtins.open', side_effect=side_effect):
            # Mock TMDB poster fetching
            mock_get_posters.return_value = {
                1: "https://image.tmdb.org/t/p/w500/poster1.jpg",
                2: "https://image.tmdb.org/t/p/w500/poster2.jpg"
            }
            
            await data_loader.load_data()
            
            assert data_loader.data_loaded is True
            assert len(data_loader.movies_cache) == 2
            assert len(data_loader.initial_movies_cache) == 2
            
            # Check movie data
            movie1 = data_loader.movies_cache[1]
            assert movie1.title == "Toy Story"
            assert movie1.year == 1995
            assert "Animation" in movie1.genres
            
            # Check initial movie data
            initial_movie1 = data_loader.initial_movies_cache[0]
            assert initial_movie1.title == "Toy Story"
            assert initial_movie1.poster_url == "https://image.tmdb.org/t/p/w500/poster1.jpg"
            assert initial_movie1.average_rating == 3.5  # Default rating from movies data
    
    @pytest.mark.asyncio
    async def test_get_initial_movies(self, data_loader):
        """Test getting initial movies."""
        # Load placeholder data
        await data_loader.load_data()
        
        initial_movies = await data_loader.get_initial_movies()
        
        assert isinstance(initial_movies, list)
        assert len(initial_movies) == 30
        assert all(isinstance(movie, MovieSummary) for movie in initial_movies)
    
    @pytest.mark.asyncio
    async def test_get_movie_by_id(self, data_loader):
        """Test getting movie by ID."""
        # Load placeholder data
        await data_loader.load_data()
        
        movie = await data_loader.get_movie_by_id(1)
        
        assert isinstance(movie, Movie)
        assert movie.id == 1
        assert movie.title is not None
        assert movie.genres is not None
    
    @pytest.mark.asyncio
    async def test_get_movie_by_invalid_id(self, data_loader):
        """Test getting movie with invalid ID."""
        await data_loader.load_data()
        
        movie = await data_loader.get_movie_by_id(99999)
        
        assert movie is None
    
    @pytest.mark.asyncio
    async def test_get_all_movies(self, data_loader):
        """Test getting all movies."""
        await data_loader.load_data()
        
        all_movies = await data_loader.get_all_movies()
        
        assert isinstance(all_movies, dict)
        assert len(all_movies) == 30  # Placeholder data
        assert all(isinstance(movie_id, int) for movie_id in all_movies.keys())
        assert all(isinstance(movie, Movie) for movie in all_movies.values())
    
    @pytest.mark.asyncio
    async def test_get_data_summary(self, data_loader):
        """Test getting data summary."""
        await data_loader.load_data()
        
        summary = data_loader.get_data_summary()
        
        assert isinstance(summary, dict)
        assert "total_movies" in summary
        assert "initial_movies" in summary
        assert "data_loaded" in summary
        assert summary["data_loaded"] is True
    
    @pytest.mark.asyncio
    async def test_get_data_summary_before_loading(self, data_loader):
        """Test getting data summary before data is loaded."""
        summary = data_loader.get_data_summary()
        
        assert isinstance(summary, dict)
        assert "error" in summary
    
    @pytest.mark.asyncio
    @patch('app.services.tmdb_service.TMDBService.get_movie_poster_url')
    async def test_movie_poster_fetching_on_demand(self, mock_get_poster, data_loader):
        """Test that movie posters are fetched on demand."""
        # Load placeholder data
        await data_loader.load_data()
        
        # Mock poster fetching
        mock_get_poster.return_value = "https://image.tmdb.org/t/p/w500/test_poster.jpg"
        
        # Get a movie (should trigger poster fetching if not present)
        movie = await data_loader.get_movie_by_id(1)
        
        assert movie is not None
        # Note: In placeholder data, posters are already set, so this tests the fallback mechanism
    
    @pytest.mark.asyncio
    async def test_concurrent_data_loading(self, data_loader):
        """Test that concurrent data loading works correctly."""
        # Create multiple tasks that try to load data
        tasks = [data_loader.load_data() for _ in range(5)]
        
        # Execute concurrently
        await asyncio.gather(*tasks)
        
        # Data should be loaded correctly
        assert data_loader.data_loaded is True
        assert data_loader.movies_cache is not None
        assert data_loader.initial_movies_cache is not None