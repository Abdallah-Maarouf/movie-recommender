"""
TMDB API integration service for fetching movie posters and metadata.
"""

import asyncio
import logging
from typing import Optional, Dict, Any
import httpx
from urllib.parse import quote
import json
from pathlib import Path

from app.core.config import settings

logger = logging.getLogger(__name__)


class TMDBService:
    """Service for interacting with The Movie Database (TMDB) API."""
    
    def __init__(self):
        self.api_key = settings.TMDB_API_KEY
        self.base_url = settings.TMDB_BASE_URL
        self.image_base_url = settings.TMDB_IMAGE_BASE_URL
        self.cache_file = Path("data/tmdb_cache.json")
        self.poster_cache: Dict[str, Optional[str]] = {}
        self.rate_limit_delay = 0.25  # 4 requests per second (TMDB limit is 40/10s)
        self._load_cache()
    
    def _load_cache(self):
        """Load cached poster URLs from disk."""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    self.poster_cache = json.load(f)
                logger.info(f"Loaded {len(self.poster_cache)} cached poster URLs")
        except Exception as e:
            logger.warning(f"Could not load TMDB cache: {e}")
            self.poster_cache = {}
    
    def _save_cache(self):
        """Save poster cache to disk."""
        try:
            self.cache_file.parent.mkdir(exist_ok=True)
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.poster_cache, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save TMDB cache: {e}")
    
    def _get_cache_key(self, title: str, year: int) -> str:
        """Generate cache key for movie."""
        return f"{title.lower().strip()}_{year}"
    
    async def get_movie_poster_url(self, title: str, year: int) -> Optional[str]:
        """
        Get movie poster URL from TMDB API with caching and rate limiting.
        
        Args:
            title: Movie title
            year: Release year
            
        Returns:
            Poster URL or None if not found
        """
        cache_key = self._get_cache_key(title, year)
        
        # Check cache first
        if cache_key in self.poster_cache:
            return self.poster_cache[cache_key]
        
        # If no API key, return None and cache the result
        if not self.api_key:
            logger.debug(f"No TMDB API key configured, skipping poster fetch for {title}")
            self.poster_cache[cache_key] = None
            return None
        
        try:
            # Rate limiting
            await asyncio.sleep(self.rate_limit_delay)
            
            # Search for movie
            poster_url = await self._search_movie_poster(title, year)
            
            # Cache the result (even if None)
            self.poster_cache[cache_key] = poster_url
            
            # Save cache periodically
            if len(self.poster_cache) % 10 == 0:
                self._save_cache()
            
            return poster_url
            
        except Exception as e:
            logger.warning(f"Error fetching poster for {title} ({year}): {e}")
            # Cache the failure to avoid repeated API calls
            self.poster_cache[cache_key] = None
            return None
    
    async def _search_movie_poster(self, title: str, year: int) -> Optional[str]:
        """Search for movie poster using TMDB API."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Clean title for search
            search_title = self._clean_title_for_search(title)
            
            # Search for movie
            search_url = f"{self.base_url}/search/movie"
            params = {
                "api_key": self.api_key,
                "query": search_title,
                "year": year,
                "include_adult": "false"
            }
            
            response = await client.get(search_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            results = data.get("results", [])
            
            if not results:
                logger.debug(f"No TMDB results found for {title} ({year})")
                return None
            
            # Find best match
            best_match = self._find_best_match(results, title, year)
            
            if best_match and best_match.get("poster_path"):
                poster_url = f"{self.image_base_url}{best_match['poster_path']}"
                logger.debug(f"Found poster for {title}: {poster_url}")
                return poster_url
            
            logger.debug(f"No poster found for {title} ({year})")
            return None
    
    def _clean_title_for_search(self, title: str) -> str:
        """Clean movie title for better TMDB search results."""
        # Remove common suffixes and prefixes
        title = title.strip()
        
        # Remove "The" prefix for better matching
        if title.lower().startswith("the "):
            title = title[4:]
        
        # Remove special characters that might interfere with search
        title = title.replace(":", "").replace("&", "and")
        
        return title.strip()
    
    def _find_best_match(self, results: list, original_title: str, year: int) -> Optional[Dict[str, Any]]:
        """Find the best matching movie from TMDB search results."""
        if not results:
            return None
        
        # First, try to find exact year match
        for result in results:
            release_date = result.get("release_date", "")
            if release_date and release_date.startswith(str(year)):
                return result
        
        # If no exact year match, try within 1 year
        for result in results:
            release_date = result.get("release_date", "")
            if release_date:
                try:
                    result_year = int(release_date[:4])
                    if abs(result_year - year) <= 1:
                        return result
                except (ValueError, IndexError):
                    continue
        
        # If still no match, return the first result
        return results[0]
    
    async def get_multiple_posters(self, movies: list) -> Dict[int, Optional[str]]:
        """
        Get poster URLs for multiple movies with batch processing and rate limiting.
        
        Args:
            movies: List of movie dictionaries with 'id', 'title', and 'year'
            
        Returns:
            Dictionary mapping movie IDs to poster URLs
        """
        poster_urls = {}
        
        # Process movies in batches to respect rate limits
        batch_size = 5
        for i in range(0, len(movies), batch_size):
            batch = movies[i:i + batch_size]
            
            # Create tasks for concurrent processing
            tasks = []
            for movie in batch:
                task = self.get_movie_poster_url(movie['title'], movie['year'])
                tasks.append((movie['id'], task))
            
            # Execute batch
            for movie_id, task in tasks:
                poster_url = await task
                poster_urls[movie_id] = poster_url
            
            # Small delay between batches
            if i + batch_size < len(movies):
                await asyncio.sleep(0.5)
        
        # Save cache after batch processing
        self._save_cache()
        
        return poster_urls
    
    def get_placeholder_poster_url(self) -> str:
        """Get placeholder poster URL for movies without posters."""
        return "https://via.placeholder.com/500x750/cccccc/666666?text=No+Poster"


# Global TMDB service instance
_tmdb_service: Optional[TMDBService] = None


def get_tmdb_service() -> TMDBService:
    """Dependency to get TMDB service instance."""
    global _tmdb_service
    
    if _tmdb_service is None:
        _tmdb_service = TMDBService()
    
    return _tmdb_service