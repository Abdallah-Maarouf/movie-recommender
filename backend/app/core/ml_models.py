"""
ML model loading and inference management.
"""

import logging
import json
import numpy as np
from typing import Dict, Optional, List, Any
import asyncio
from pathlib import Path
import joblib

from app.models.recommendation import RecommendationResponse, RecommendationItem
from app.models.movie import MovieSummary
from app.core.config import settings

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages loading and inference of ML models for recommendations."""
    
    def __init__(self):
        self.models_loaded = False
        self.svd_model = None
        self.item_similarity_matrix = None
        self.content_similarity_matrix = None
        self.hybrid_config = None
        self.fallback_recommendations = None
        self.movies_data = None
        self.movie_id_to_index = {}
        self.index_to_movie_id = {}
        
    async def load_models(self):
        """Load all ML models from disk."""
        if self.models_loaded:
            return
            
        try:
            logger.info("Loading ML models...")
            models_dir = Path(settings.MODELS_DIR)
            data_dir = Path(settings.DATA_DIR)
            
            if not models_dir.exists():
                logger.warning(f"Models directory not found: {models_dir}")
                await self._load_fallback_data(models_dir)
                return
            
            await self._load_movie_data(data_dir)
            await self._load_collaborative_models(models_dir)
            await self._load_content_models(models_dir)
            await self._load_hybrid_config(models_dir)
            await self._load_fallback_data(models_dir)
            
            self.models_loaded = True
            logger.info("All ML models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            await self._load_fallback_data(models_dir)
            logger.warning("Using fallback recommendations only")
    
    async def _load_movie_data(self, data_dir: Path):
        """Load movie metadata."""
        movies_file = data_dir / "movies.json"
        if movies_file.exists():
            with open(movies_file, 'r', encoding='utf-8') as f:
                self.movies_data = json.load(f)
            
            for idx, movie in enumerate(self.movies_data):
                movie_id = movie.get('id', movie.get('MovieID'))  # Handle both formats
                self.movie_id_to_index[movie_id] = idx
                self.index_to_movie_id[idx] = movie_id
                
            logger.info(f"Loaded {len(self.movies_data)} movies")
        else:
            logger.warning("Movies data file not found")
    
    async def _load_collaborative_models(self, models_dir: Path):
        """Load collaborative filtering models."""
        try:
            svd_file = models_dir / "collaborative_svd_model.pkl"
            if svd_file.exists():
                self.svd_model = joblib.load(svd_file)
                logger.info("Loaded SVD model")
            
            item_sim_file = models_dir / "item_similarity_matrix.pkl"
            if item_sim_file.exists():
                self.item_similarity_matrix = joblib.load(item_sim_file)
                logger.info("Loaded item similarity matrix")
                
        except Exception as e:
            logger.warning(f"Error loading collaborative models: {e}")
    
    async def _load_content_models(self, models_dir: Path):
        """Load content-based filtering models."""
        try:
            content_sim_file = models_dir / "content_similarity_matrix.pkl"
            if content_sim_file.exists():
                self.content_similarity_matrix = joblib.load(content_sim_file)
                logger.info("Loaded content similarity matrix")
                
        except Exception as e:
            logger.warning(f"Error loading content models: {e}")
    
    async def _load_hybrid_config(self, models_dir: Path):
        """Load hybrid model configuration."""
        try:
            config_file = models_dir / "hybrid_model_config.json"
            if config_file.exists():
                with open(config_file, 'r') as f:
                    self.hybrid_config = json.load(f)
                logger.info("Loaded hybrid model configuration")
        except Exception as e:
            logger.warning(f"Error loading hybrid config: {e}")
    
    async def _load_fallback_data(self, models_dir: Path):
        """Load fallback recommendations."""
        try:
            fallback_file = models_dir / "fallback_recommendations.json"
            if fallback_file.exists():
                with open(fallback_file, 'r') as f:
                    self.fallback_recommendations = json.load(f)
                logger.info(f"Loaded {len(self.fallback_recommendations)} fallback recommendations")
        except Exception as e:
            logger.warning(f"Error loading fallback recommendations: {e}")
            self.fallback_recommendations = []
    
    def is_ready(self) -> bool:
        """Check if models are loaded and ready."""
        return self.models_loaded or (self.fallback_recommendations is not None)
    
    def _validate_models(self) -> Dict[str, bool]:
        """Validate model availability."""
        return {
            'collaborative_available': self.item_similarity_matrix is not None,
            'content_available': self.content_similarity_matrix is not None,
            'hybrid_available': self.hybrid_config is not None,
            'fallback_available': self.fallback_recommendations is not None,
            'movies_available': self.movies_data is not None
        }
    
    async def generate_recommendations(
        self,
        ratings: Dict[int, float],
        algorithm: str = "hybrid",
        num_recommendations: int = 20
    ) -> RecommendationResponse:
        """Generate recommendations based on user ratings."""
        import time
        start_time = time.time()
        
        logger.info(f"Generating recommendations using {algorithm} algorithm")
        
        if len(ratings) < settings.MIN_RATINGS_REQUIRED:
            raise ValueError(f"At least {settings.MIN_RATINGS_REQUIRED} ratings required")
        
        model_status = self._validate_models()
        
        try:
            if algorithm == "collaborative" and model_status['collaborative_available']:
                recommendations = await self._generate_collaborative_recommendations(ratings, num_recommendations)
            elif algorithm == "content" and model_status['content_available']:
                recommendations = await self._generate_content_recommendations(ratings, num_recommendations)
            elif algorithm == "hybrid" and model_status['hybrid_available']:
                recommendations = await self._generate_hybrid_recommendations(ratings, num_recommendations)
            else:
                logger.warning(f"Algorithm '{algorithm}' not available, using fallback")
                recommendations = await self._generate_fallback_recommendations(ratings, num_recommendations)
                algorithm = "fallback"
            
            processing_time = time.time() - start_time
            
            confidence_score = sum(rec.confidence for rec in recommendations) / len(recommendations) if recommendations else 0.0
            
            return RecommendationResponse(
                recommendations=recommendations,
                total_ratings=len(ratings),
                algorithm_used=algorithm,
                confidence_score=confidence_score,
                processing_time=processing_time,
                metadata={"model_status": model_status, "fallback_used": algorithm == "fallback"}
            )
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            recommendations = await self._generate_fallback_recommendations(ratings, num_recommendations)
            processing_time = time.time() - start_time
            
            return RecommendationResponse(
                recommendations=recommendations,
                total_ratings=len(ratings),
                algorithm_used="fallback",
                confidence_score=0.5,
                processing_time=processing_time,
                metadata={"error": str(e), "fallback_used": True}
            )

    async def _generate_collaborative_recommendations(
        self, 
        ratings: Dict[int, float], 
        num_recommendations: int
    ) -> List[RecommendationItem]:
        """Generate recommendations using collaborative filtering."""
        logger.info("Generating collaborative filtering recommendations")
        
        if self.item_similarity_matrix is None or self.movies_data is None:
            raise ValueError("Collaborative filtering models not available")
        
        user_ratings = np.zeros(len(self.movies_data))
        rated_indices = []
        
        for movie_id, rating in ratings.items():
            if movie_id in self.movie_id_to_index:
                idx = self.movie_id_to_index[movie_id]
                user_ratings[idx] = rating
                rated_indices.append(idx)
        
        if len(rated_indices) == 0:
            raise ValueError("No valid movie IDs found in ratings")
        
        predictions = np.zeros(len(self.movies_data))
        
        for i in range(len(self.movies_data)):
            if i in rated_indices:
                continue
            
            similarities = self.item_similarity_matrix[i]
            weighted_sum = 0.0
            similarity_sum = 0.0
            
            for j in rated_indices:
                if similarities[j] > 0.1:
                    weighted_sum += similarities[j] * user_ratings[j]
                    similarity_sum += similarities[j]
            
            if similarity_sum > 0:
                predictions[i] = weighted_sum / similarity_sum
        
        top_indices = np.argsort(predictions)[::-1][:num_recommendations * 2]
        
        recommendations = []
        for idx in top_indices:
            if predictions[idx] > 0 and len(recommendations) < num_recommendations:
                movie_id = self.index_to_movie_id[idx]
                movie_data = self.movies_data[idx]
                
                similarities = self.item_similarity_matrix[idx]
                similar_rated_count = sum(1 for j in rated_indices if similarities[j] > 0.1)
                confidence = min(0.9, similar_rated_count / 10.0)
                
                explanation = self._generate_collaborative_explanation(idx, rated_indices, ratings)
                
                recommendations.append(RecommendationItem(
                    movie=self._create_movie_summary(movie_data),
                    predicted_rating=min(5.0, max(1.0, predictions[idx])),
                    confidence=confidence,
                    explanation=explanation,
                    algorithm_used="collaborative"
                ))
        
        return recommendations

    async def _generate_content_recommendations(
        self, 
        ratings: Dict[int, float], 
        num_recommendations: int
    ) -> List[RecommendationItem]:
        """Generate recommendations using content-based filtering."""
        logger.info("Generating content-based recommendations")
        
        if self.content_similarity_matrix is None or self.movies_data is None:
            raise ValueError("Content-based models not available")
        
        user_profile = np.zeros(self.content_similarity_matrix.shape[0])
        rated_indices = []
        
        for movie_id, rating in ratings.items():
            if movie_id in self.movie_id_to_index:
                idx = self.movie_id_to_index[movie_id]
                weight = (rating - 3.0) / 2.0
                user_profile += weight * self.content_similarity_matrix[idx]
                rated_indices.append(idx)
        
        if len(rated_indices) == 0:
            raise ValueError("No valid movie IDs found in ratings")
        
        user_profile = user_profile / len(rated_indices)
        content_scores = np.dot(self.content_similarity_matrix, user_profile)
        
        for idx in rated_indices:
            content_scores[idx] = 0
        
        top_indices = np.argsort(content_scores)[::-1][:num_recommendations]
        
        recommendations = []
        for idx in top_indices:
            if content_scores[idx] > 0:
                movie_id = self.index_to_movie_id[idx]
                movie_data = self.movies_data[idx]
                
                confidence = min(0.9, content_scores[idx])
                explanation = self._generate_content_explanation(idx, rated_indices, ratings)
                
                avg_user_rating = sum(ratings.values()) / len(ratings)
                predicted_rating = avg_user_rating + (content_scores[idx] - 0.5) * 2
                predicted_rating = min(5.0, max(1.0, predicted_rating))
                
                recommendations.append(RecommendationItem(
                    movie=self._create_movie_summary(movie_data),
                    predicted_rating=predicted_rating,
                    confidence=confidence,
                    explanation=explanation,
                    algorithm_used="content"
                ))
        
        return recommendations

    async def _generate_hybrid_recommendations(
        self, 
        ratings: Dict[int, float], 
        num_recommendations: int
    ) -> List[RecommendationItem]:
        """Generate recommendations using hybrid approach."""
        logger.info("Generating hybrid recommendations")
        
        model_status = self._validate_models()
        
        if not (model_status['collaborative_available'] and model_status['content_available']):
            if model_status['collaborative_available']:
                return await self._generate_collaborative_recommendations(ratings, num_recommendations)
            elif model_status['content_available']:
                return await self._generate_content_recommendations(ratings, num_recommendations)
            else:
                raise ValueError("Neither collaborative nor content models available")
        
        collab_recs = await self._generate_collaborative_recommendations(ratings, num_recommendations * 2)
        content_recs = await self._generate_content_recommendations(ratings, num_recommendations * 2)
        
        alpha = self.hybrid_config.get('best_alpha', 0.7) if self.hybrid_config else 0.7
        
        movie_scores = {}
        
        for rec in collab_recs:
            movie_id = rec.movie.id
            movie_scores[movie_id] = {
                'collab_score': rec.predicted_rating * rec.confidence,
                'collab_confidence': rec.confidence,
                'collab_explanation': rec.explanation,
                'movie_data': rec.movie
            }
        
        for rec in content_recs:
            movie_id = rec.movie.id
            if movie_id in movie_scores:
                movie_scores[movie_id]['content_score'] = rec.predicted_rating * rec.confidence
                movie_scores[movie_id]['content_confidence'] = rec.confidence
                movie_scores[movie_id]['content_explanation'] = rec.explanation
            else:
                movie_scores[movie_id] = {
                    'collab_score': 0,
                    'collab_confidence': 0,
                    'collab_explanation': "",
                    'content_score': rec.predicted_rating * rec.confidence,
                    'content_confidence': rec.confidence,
                    'content_explanation': rec.explanation,
                    'movie_data': rec.movie
                }
        
        hybrid_recommendations = []
        for movie_id, scores in movie_scores.items():
            collab_score = scores.get('collab_score', 0)
            content_score = scores.get('content_score', 0)
            
            hybrid_score = alpha * collab_score + (1 - alpha) * content_score
            
            if hybrid_score > 0:
                collab_conf = scores.get('collab_confidence', 0)
                content_conf = scores.get('content_confidence', 0)
                hybrid_confidence = alpha * collab_conf + (1 - alpha) * content_conf
                
                explanations = []
                if scores.get('collab_explanation'):
                    explanations.append(f"Collaborative: {scores['collab_explanation']}")
                if scores.get('content_explanation'):
                    explanations.append(f"Content: {scores['content_explanation']}")
                
                hybrid_explanation = "; ".join(explanations) if explanations else "Hybrid recommendation"
                
                hybrid_recommendations.append({
                    'movie_id': movie_id,
                    'score': hybrid_score,
                    'confidence': hybrid_confidence,
                    'explanation': hybrid_explanation,
                    'movie_data': scores['movie_data']
                })
        
        hybrid_recommendations.sort(key=lambda x: x['score'], reverse=True)
        hybrid_recommendations = hybrid_recommendations[:num_recommendations]
        
        recommendations = []
        for rec in hybrid_recommendations:
            recommendations.append(RecommendationItem(
                movie=rec['movie_data'],
                predicted_rating=min(5.0, max(1.0, rec['score'])),
                confidence=rec['confidence'],
                explanation=rec['explanation'],
                algorithm_used="hybrid"
            ))
        
        return recommendations

    async def _generate_fallback_recommendations(
        self, 
        ratings: Dict[int, float], 
        num_recommendations: int
    ) -> List[RecommendationItem]:
        """Generate fallback recommendations using popular movies."""
        logger.info("Generating fallback recommendations")
        
        if not self.fallback_recommendations:
            raise ValueError("No fallback recommendations available")
        
        rated_movie_ids = set(ratings.keys())
        available_fallbacks = [
            rec for rec in self.fallback_recommendations 
            if rec['movie_id'] not in rated_movie_ids
        ]
        
        selected_fallbacks = available_fallbacks[:num_recommendations]
        
        recommendations = []
        for fallback in selected_fallbacks:
            movie_data = None
            if self.movies_data:
                for movie in self.movies_data:
                    movie_id = movie.get('id', movie.get('MovieID'))
                    if movie_id == fallback['movie_id']:
                        movie_data = movie
                        break
            
            if movie_data:
                movie_summary = self._create_movie_summary(movie_data)
            else:
                movie_summary = MovieSummary(
                    id=fallback['movie_id'],
                    title=fallback['title'],
                    genres=fallback['genres'].split('|') if '|' in fallback['genres'] else [fallback['genres']],
                    year=1995,
                    poster_url=None,
                    average_rating=fallback['avg_rating']
                )
            
            recommendations.append(RecommendationItem(
                movie=movie_summary,
                predicted_rating=fallback['avg_rating'],
                confidence=0.6,
                explanation=fallback['explanation'],
                algorithm_used="fallback"
            ))
        
        return recommendations
    
    def _create_movie_summary(self, movie_data: Dict) -> MovieSummary:
        """Create MovieSummary from movie data."""
        # Handle different field name formats
        movie_id = movie_data.get('id', movie_data.get('MovieID'))
        title = movie_data.get('title', movie_data.get('Title', movie_data.get('CleanTitle', 'Unknown')))
        genres = movie_data.get('genres', movie_data.get('GenreList', []))
        year = movie_data.get('year', movie_data.get('Year', 1995))
        
        # Convert genres string to list if needed
        if isinstance(genres, str):
            genres = genres.split('|')
        
        return MovieSummary(
            id=movie_id,
            title=title,
            genres=genres,
            year=year,
            poster_url=movie_data.get('poster_url'),
            average_rating=movie_data.get('average_rating', 3.5)
        )
    
    def _generate_collaborative_explanation(
        self, 
        movie_idx: int, 
        rated_indices: List[int], 
        ratings: Dict[int, float]
    ) -> str:
        """Generate explanation for collaborative filtering recommendation."""
        if self.item_similarity_matrix is None:
            return "Based on collaborative filtering"
        
        similarities = self.item_similarity_matrix[movie_idx]
        similar_movies = []
        
        for rated_idx in rated_indices:
            if similarities[rated_idx] > 0.3:
                movie_id = self.index_to_movie_id[rated_idx]
                movie_title = self.movies_data[rated_idx]['title']
                rating = ratings[movie_id]
                similar_movies.append((movie_title, rating, similarities[rated_idx]))
        
        similar_movies.sort(key=lambda x: x[2], reverse=True)
        similar_movies = similar_movies[:3]
        
        if similar_movies:
            movie_names = [movie[0] for movie in similar_movies]
            if len(movie_names) == 1:
                return f"Because you liked {movie_names[0]}"
            elif len(movie_names) == 2:
                return f"Because you liked {movie_names[0]} and {movie_names[1]}"
            else:
                return f"Because you liked {', '.join(movie_names[:-1])}, and {movie_names[-1]}"
        
        return "Based on your rating patterns"
    
    def _generate_content_explanation(
        self, 
        movie_idx: int, 
        rated_indices: List[int], 
        ratings: Dict[int, float]
    ) -> str:
        """Generate explanation for content-based recommendation."""
        if self.content_similarity_matrix is None:
            return "Based on movie content similarity"
        
        similarities = self.content_similarity_matrix[movie_idx]
        similar_movies = []
        
        for rated_idx in rated_indices:
            if similarities[rated_idx] > 0.3:
                movie_id = self.index_to_movie_id[rated_idx]
                movie_title = self.movies_data[rated_idx]['title']
                rating = ratings[movie_id]
                if rating >= 4.0:
                    similar_movies.append((movie_title, rating, similarities[rated_idx]))
        
        similar_movies.sort(key=lambda x: x[2], reverse=True)
        similar_movies = similar_movies[:2]
        
        if similar_movies:
            movie_names = [movie[0] for movie in similar_movies]
            if len(movie_names) == 1:
                return f"Similar content to {movie_names[0]} which you rated highly"
            else:
                return f"Similar content to {' and '.join(movie_names)} which you rated highly"
        
        return "Based on content similarity to your preferences"


# Global model manager instance
_model_manager: Optional[ModelManager] = None


async def get_model_manager() -> ModelManager:
    """Dependency to get model manager instance."""
    global _model_manager
    
    if _model_manager is None:
        _model_manager = ModelManager()
        await _model_manager.load_models()
    
    return _model_manager