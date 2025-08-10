"""
Pytest configuration and fixtures for the Movie Recommendation System API tests.
"""

import pytest
import asyncio
from httpx import AsyncClient
from fastapi.testclient import TestClient

from app.main import app
from app.utils.data_loader import DataLoader
from app.core.ml_models import ModelManager


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
async def async_client():
    """Create an async test client for the FastAPI app."""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac


@pytest.fixture
async def data_loader():
    """Create a test data loader instance."""
    loader = DataLoader()
    await loader.load_data()
    return loader


@pytest.fixture
async def model_manager():
    """Create a test model manager instance."""
    manager = ModelManager()
    await manager.load_models()
    return manager


@pytest.fixture
def sample_ratings():
    """Sample user ratings for testing."""
    return {
        1: 4.0,
        2: 3.5,
        3: 5.0,
        4: 2.0,
        5: 4.5,
        6: 3.0,
        7: 4.0,
        8: 3.5,
        9: 5.0,
        10: 2.5,
        11: 4.0,
        12: 3.0,
        13: 4.5,
        14: 3.5,
        15: 4.0,
        16: 3.0,
        17: 4.5,
        18: 3.5
    }


@pytest.fixture
def sample_movie_data():
    """Sample movie data for testing."""
    return {
        "id": 1,
        "title": "Test Movie",
        "genres": ["Action", "Drama"],
        "year": 2020,
        "average_rating": 4.2,
        "rating_count": 1500,
        "poster_url": "https://example.com/poster.jpg",
        "description": "A test movie for unit testing",
        "director": "Test Director",
        "cast": ["Actor 1", "Actor 2"],
        "runtime": 120
    }