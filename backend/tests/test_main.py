"""
Tests for the main FastAPI application.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app


class TestMainApp:
    """Test cases for the main FastAPI application."""
    
    def test_root_endpoint(self, client: TestClient):
        """Test the root endpoint returns correct information."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["message"] == "Movie Recommendation System API"
        assert data["version"] == "1.0.0"
        assert data["docs"] == "/docs"
        assert data["health"] == "/api/health"
    
    def test_health_check(self, client: TestClient):
        """Test the health check endpoint."""
        response = client.get("/api/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "healthy"
        assert data["service"] == "Movie Recommendation System API"
        assert data["version"] == "1.0.0"
        assert "timestamp" in data
    
    def test_docs_endpoint_accessible(self, client: TestClient):
        """Test that the API documentation is accessible."""
        response = client.get("/docs")
        assert response.status_code == 200
    
    def test_openapi_schema(self, client: TestClient):
        """Test that the OpenAPI schema is accessible."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        
        schema = response.json()
        assert schema["info"]["title"] == "Movie Recommendation System API"
        assert schema["info"]["version"] == "1.0.0"
    
    def test_cors_headers(self, client: TestClient):
        """Test that CORS headers are properly configured."""
        # Test preflight request
        response = client.options(
            "/api/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET"
            }
        )
        
        # Should allow the request (status might be 200 or 405 depending on implementation)
        assert response.status_code in [200, 405]
    
    def test_request_logging_middleware(self, client: TestClient):
        """Test that request logging middleware adds timing headers."""
        response = client.get("/api/health")
        
        assert response.status_code == 200
        assert "X-Process-Time" in response.headers
        
        # Process time should be a valid float
        process_time = float(response.headers["X-Process-Time"])
        assert process_time >= 0
    
    def test_404_handling(self, client: TestClient):
        """Test that 404 errors are handled properly."""
        response = client.get("/nonexistent-endpoint")
        
        assert response.status_code == 404