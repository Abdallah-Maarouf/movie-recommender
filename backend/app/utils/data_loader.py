"""
Data loading utilities for movies and ratings with TMDB integration.
"""

import json
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
import asyncio
import pandas as pd

from app.models.movie import Movie, MovieSummary
from app.core.config import settings
from app.services.tmdb_service import get_tmdb_service

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Handles loading and caching of movie data with TMDB poster integration.
    """
    
    def __init__(self):
        self.movies_cache: Optional[Dict[int, Movie]] = None
        self.initial_movies_cache: Optional[List[MovieSummary]] = None
        self.all_movies_data: Optional[List[Dict]] = None
        self.data_loaded = False
        self.tmdb_service = get_tmdb_service()
        
    async def load_data(self):
        """Load all movie data from JSON files with TMDB poster integration."""
        try:
            logger.info("Loading movie data from JSON files...")
            
            data_dir = Path(settings.DATA_DIR)
            
            # Check if data directory exists
            if not data_dir.exists():
                logger.warning(f"Data directory not found: {data_dir}")
                logger.info("Creating placeholder data for development")
                await self._create_placeholder_data()
                return
            
            # Load main movies data
            movies_file = data_dir / "movies.json"
            initial_movies_file = data_dir / "initial_movies.json"
            
            if not movies_file.exists() or not initial_movies_file.exists():
                logger.warning("Required data files not found, using placeholder data")
                await self._create_placeholder_data()
                return
            
            # Load movies data
            await self._load_movies_data(movies_file)
            
            # Load initial movies data
            await self._load_initial_movies_data(initial_movies_file)
            
            # Load ratings data for movie statistics
            await self._load_ratings_statistics()
            
            self.data_loaded = True
            logger.info(f"Successfully loaded {len(self.movies_cache)} movies and {len(self.initial_movies_cache)} initial movies")
            
        except Exception as e:
            logger.error(f"Error loading data: {e}", exc_info=True)
            logger.info("Falling back to placeholder data")
            await self._create_placeholder_data()
    
    async def _load_movies_data(self, movies_file: Path):
        """Load main movies data from JSON file."""
        logger.info(f"Loading movies from {movies_file}")
        
        with open(movies_file, 'r', encoding='utf-8') as f:
            movies_data = json.load(f)
        
        self.all_movies_data = movies_data
        
        # Create movies cache
        self.movies_cache = {}
        
        # Process movies in batches for TMDB poster fetching
        logger.info("Processing movies and fetching posters...")
        
        for movie_data in movies_data:
            movie_id = movie_data['MovieID']
            title = movie_data['CleanTitle']
            year = movie_data['Year']
            genres = movie_data['GenreList']
            
            # Create Movie object (poster will be added later if needed)
            movie = Movie(
                id=movie_id,
                title=title,
                genres=genres,
                year=year,
                average_rating=3.5,  # Default, will be updated with ratings data
                rating_count=0,      # Default, will be updated with ratings data
                poster_url=None,     # Will be fetched from TMDB when needed
                description=f"A {year} {', '.join(genres[:2])} film.",
                director="Unknown",
                cast=[],
                runtime=120
            )
            
            self.movies_cache[movie_id] = movie
        
        logger.info(f"Loaded {len(self.movies_cache)} movies")
    
    async def _load_initial_movies_data(self, initial_movies_file: Path):
        """Load initial movies data with TMDB posters."""
        logger.info(f"Loading initial movies from {initial_movies_file}")
        
        with open(initial_movies_file, 'r', encoding='utf-8') as f:
            initial_movies_data = json.load(f)
        
        # Prepare movies for poster fetching
        movies_for_posters = []
        for movie_data in initial_movies_data:
            movies_for_posters.append({
                'id': movie_data['MovieID'],
                'title': movie_data['CleanTitle'],
                'year': movie_data['Year']
            })
        
        # Fetch posters for initial movies
        logger.info("Fetching posters for initial movies from TMDB...")
        poster_urls = await self.tmdb_service.get_multiple_posters(movies_for_posters)
        
        # Create MovieSummary objects
        self.initial_movies_cache = []
        
        for movie_data in initial_movies_data:
            movie_id = movie_data['MovieID']
            poster_url = poster_urls.get(movie_id)
            
            # Use placeholder if no poster found
            if not poster_url:
                poster_url = self.tmdb_service.get_placeholder_poster_url()
            
            movie_summary = MovieSummary(
                id=movie_id,
                title=movie_data['CleanTitle'],
                genres=movie_data['GenreList'],
                year=movie_data['Year'],
                poster_url=poster_url,
                average_rating=round(movie_data.get('avg_rating', 3.5), 1)
            )
            
            self.initial_movies_cache.append(movie_summary)
            
            # Update the full movie cache with poster and rating info
            if movie_id in self.movies_cache:
                self.movies_cache[movie_id].poster_url = poster_url
                self.movies_cache[movie_id].average_rating = round(movie_data.get('avg_rating', 3.5), 1)
                self.movies_cache[movie_id].rating_count = int(movie_data.get('rating_count', 0))
        
        logger.info(f"Loaded {len(self.initial_movies_cache)} initial movies with posters")
    
    async def _load_ratings_statistics(self):
        """Load ratings statistics from CSV file if available."""
        try:
            data_dir = Path(settings.DATA_DIR)
            ratings_file = data_dir / "ratings.csv"
            
            if not ratings_file.exists():
                logger.info("Ratings file not found, using default statistics")
                return
            
            logger.info("Loading ratings statistics...")
            
            # Read ratings data
            ratings_df = pd.read_csv(ratings_file)
            
            # Calculate statistics per movie
            movie_stats = ratings_df.groupby('MovieID').agg({
                'Rating': ['mean', 'count']
            }).round(1)
            
            movie_stats.columns = ['avg_rating', 'rating_count']
            
            # Update movie cache with real statistics
            for movie_id, stats in movie_stats.iterrows():
                if movie_id in self.movies_cache:
                    self.movies_cache[movie_id].average_rating = float(stats['avg_rating'])
                    self.movies_cache[movie_id].rating_count = int(stats['rating_count'])
            
            logger.info(f"Updated ratings statistics for {len(movie_stats)} movies")
            
        except Exception as e:
            logger.warning(f"Could not load ratings statistics: {e}")
    
    async def _create_placeholder_data(self):
        """Create placeholder data for development."""
        logger.info("Creating placeholder movie data")
        
        # Create placeholder movies
        placeholder_movies = []
        for i in range(30):
            movie = MovieSummary(
                id=i + 1,
                title=f"Placeholder Movie {i + 1}",
                genres=["Drama", "Action"] if i % 2 == 0 else ["Comedy", "Romance"],
                year=2000 + (i % 24),
                poster_url=self.tmdb_service.get_placeholder_poster_url(),
                average_rating=3.0 + (i % 3)
            )
            placeholder_movies.append(movie)
        
        self.initial_movies_cache = placeholder_movies
        
        # Create movies cache
        self.movies_cache = {}
        for movie in placeholder_movies:
            full_movie = Movie(
                id=movie.id,
                title=movie.title,
                genres=movie.genres,
                year=movie.year,
                average_rating=movie.average_rating,
                rating_count=100 + (movie.id * 10),
                poster_url=movie.poster_url,
                description=f"Placeholder description for {movie.title}",
                director="Placeholder Director",
                cast=["Actor 1", "Actor 2"],
                runtime=120
            )
            self.movies_cache[movie.id] = full_movie
        
        self.data_loaded = True
        logger.info(f"Created {len(placeholder_movies)} placeholder movies")
    
    async def get_initial_movies(self) -> List[MovieSummary]:
        """Get the initial 30 movies for rating interface."""
        if not self.data_loaded:
            await self.load_data()
        
        if self.initial_movies_cache is None:
            raise ValueError("Initial movies data not available")
        
        return self.initial_movies_cache
    
    async def get_movie_by_id(self, movie_id: int) -> Optional[Movie]:
        """Get detailed movie information by ID."""
        if not self.data_loaded:
            await self.load_data()
        
        if self.movies_cache is None:
            return None
        
        movie = self.movies_cache.get(movie_id)
        
        # If movie exists but doesn't have a poster, try to fetch it
        if movie and not movie.poster_url:
            poster_url = await self.tmdb_service.get_movie_poster_url(movie.title, movie.year)
            if poster_url:
                movie.poster_url = poster_url
            else:
                movie.poster_url = self.tmdb_service.get_placeholder_poster_url()
        
        return movie
    
    async def get_all_movies(self) -> Dict[int, Movie]:
        """Get all movies data."""
        if not self.data_loaded:
            await self.load_data()
        
        if self.movies_cache is None:
            raise ValueError("Movies data not available")
        
        return self.movies_cache
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary statistics about the loaded data."""
        if not self.data_loaded or not self.movies_cache:
            return {"error": "Data not loaded"}
        
        return {
            "total_movies": len(self.movies_cache),
            "initial_movies": len(self.initial_movies_cache) if self.initial_movies_cache else 0,
            "movies_with_posters": sum(1 for movie in self.movies_cache.values() if movie.poster_url and "placeholder" not in movie.poster_url.lower()),
            "data_loaded": self.data_loaded
        }


# Global data loader instance
_data_loader: Optional[DataLoader] = None


async def get_data_loader() -> DataLoader:
    """Dependency to get data loader instance."""
    global _data_loader
    
    if _data_loader is None:
        _data_loader = DataLoader()
        await _data_loader.load_data()
    
    return _data_loader