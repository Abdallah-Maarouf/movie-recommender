"""
Tests for recommendation update functionality and session management integration.
"""

import pytest
import json
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from fastapi import HTTPException

from app.main import app
from app.models.recommendation import UpdateRecommendationRequest, RecommendationResponse
from app.services.recommendation_engine import RecommendationEngine
from app.services.session_manager import SessionManager


class TestRecommendationUpdateAPI:
    """Test recommendation update API endpoints."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.client = TestClient(app)
        self.base_url = "/api/recommendations"
    
    @patch('app.api.recommendations.get_recommendation_engine')
    @patch('app.api.recommendations.get_model_manager')
    async def test_update_recommendations_success(self, mock_model_manager, mock_get_engine):
        """Test successful recommendation update."""
        # Mock recommendation engine
        mock_engine = AsyncMock()
        mock_response = RecommendationResponse(
            recommendations=[],
            total_ratings=20,
            algorithm_used="hybrid",
            confidence_score=0.85,
            processing_time=0.5,
            metadata={
                'update_request': True,
                'incremental_computation_used': True
            }
        )
        mock_engine.update_recommendations.return_value = mock_response
        mock_get_engine.return_value = mock_engine
        
        # Test data
        request_data = {
            "existing_ratings": {
                "1": 4.0,
                "2": 3.5,
                "3": 5.0
            },
            "new_ratings": {
                "4": 4.0,
                "5": 3.0
            },
            "algorithm": "hybrid",
            "num_recommendations": 20
        }
        
        response = self.client.post(f"{self.base_url}/update", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data['total_ratings'] == 20
        assert data['algorithm_used'] == "hybrid"
        assert 'metadata' in data
    
    def test_update_recommendations_invalid_data(self):
        """Test update recommendations with invalid data."""
        request_data = {
            "existing_ratings": {
                "1": 6.0  # Invalid rating > 5.0
            },
            "new_ratings": {
                "2": 4.0
            }
        }
        
        response = self.client.post(f"{self.base_url}/update", json=request_data)
        
        assert response.status_code == 422  # Pydantic validation error
        data = response.json()
        assert 'detail' in data
    
    def test_update_recommendations_missing_existing_ratings(self):
        """Test update recommendations with missing existing ratings."""
        request_data = {
            "new_ratings": {
                "1": 4.0,
                "2": 3.5
            }
        }
        
        response = self.client.post(f"{self.base_url}/update", json=request_data)
        
        assert response.status_code == 422  # Validation error
    
    def test_update_recommendations_empty_new_ratings(self):
        """Test update recommendations with empty new ratings."""
        request_data = {
            "existing_ratings": {
                "1": 4.0,
                "2": 3.5
            },
            "new_ratings": {}
        }
        
        response = self.client.post(f"{self.base_url}/update", json=request_data)
        
        assert response.status_code == 422  # Validation error


class TestRecommendationUpdateLogic:
    """Test recommendation update logic and session integration."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.session_manager = SessionManager()
    
    def test_session_validation_for_updates(self):
        """Test session validation during recommendation updates."""
        # Valid session data
        session_data = {
            'ratings': {
                1: 4.0, 2: 3.5, 3: 5.0, 4: 2.0, 5: 4.5,
                6: 3.0, 7: 4.0, 8: 3.5, 9: 5.0, 10: 2.5,
                11: 4.0, 12: 3.5, 13: 5.0, 14: 2.0, 15: 4.5
            }
        }
        
        is_valid, error_msg = self.session_manager.validate_session(session_data)
        assert is_valid is True
        assert error_msg is None
    
    def test_recommendation_delta_calculation(self):
        """Test calculation of recommendation changes."""
        old_recommendations = [
            {'movie': {'id': 1, 'title': 'Movie 1'}, 'predicted_rating': 4.0},
            {'movie': {'id': 2, 'title': 'Movie 2'}, 'predicted_rating': 3.5},
            {'movie': {'id': 3, 'title': 'Movie 3'}, 'predicted_rating': 4.2}
        ]
        
        new_recommendations = [
            {'movie': {'id': 1, 'title': 'Movie 1'}, 'predicted_rating': 4.3},  # Rating changed
            {'movie': {'id': 3, 'title': 'Movie 3'}, 'predicted_rating': 4.2},  # Same
            {'movie': {'id': 4, 'title': 'Movie 4'}, 'predicted_rating': 4.5}   # New movie
        ]
        
        delta = self.session_manager.calculate_recommendation_delta(
            old_recommendations, new_recommendations
        )
        
        assert delta['new_movies'] == 1  # Movie 4 is new
        assert delta['removed_movies'] == 1  # Movie 2 was removed
        assert len(delta['rating_changes']) > 0  # Movie 1 rating changed
        assert 'change_summary' in delta
    
    def test_session_analytics_for_updates(self):
        """Test session analytics during recommendation updates."""
        ratings = {i: 3.5 + (i % 3) * 0.5 for i in range(1, 21)}
        
        analysis = self.session_manager.analyze_session(ratings, 'hybrid')
        
        assert 'session_id' in analysis
        assert 'confidence_score' in analysis
        assert 'pattern_analysis' in analysis
        assert analysis['total_ratings'] == 20
        assert analysis['algorithm_used'] == 'hybrid'
    
    def test_session_optimization_for_updates(self):
        """Test session data optimization for better performance."""
        ratings = {5: 4.0, 1: 3.5, 3: 5.0, 2: 2.0}
        
        optimizations = self.session_manager.optimize_session_data(ratings)
        
        assert 'sorted_ratings' in optimizations
        assert 'rating_vector' in optimizations
        assert 'movie_ids' in optimizations
        assert 'mean_rating' in optimizations
        
        # Check that ratings are sorted by movie ID
        sorted_ratings = optimizations['sorted_ratings']
        assert list(sorted_ratings.keys()) == [1, 2, 3, 5]


if __name__ == "__main__":
    pytest.main([__file__])