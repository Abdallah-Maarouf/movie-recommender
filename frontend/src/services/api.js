import axios from 'axios';

// API configuration
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// Create axios instance with default configuration
const api = axios.create({
    baseURL: API_BASE_URL,
    timeout: 30000, // 30 seconds timeout
    headers: {
        'Content-Type': 'application/json',
    },
});

// Request interceptor for logging and adding auth headers if needed
api.interceptors.request.use(
    (config) => {
        console.log(`Making ${config.method?.toUpperCase()} request to ${config.url}`);
        return config;
    },
    (error) => {
        console.error('Request error:', error);
        return Promise.reject(error);
    }
);

// Response interceptor for error handling and logging
api.interceptors.response.use(
    (response) => {
        console.log(`Response from ${response.config.url}:`, response.status);
        return response;
    },
    (error) => {
        console.error('Response error:', error);

        // Handle different error types
        if (error.response) {
            // Server responded with error status
            const { status, data } = error.response;
            console.error(`API Error ${status}:`, data);

            // You can add specific error handling here
            switch (status) {
                case 400:
                    console.error('Bad Request:', data.message || 'Invalid request');
                    break;
                case 404:
                    console.error('Not Found:', data.message || 'Resource not found');
                    break;
                case 500:
                    console.error('Server Error:', data.message || 'Internal server error');
                    break;
                default:
                    console.error('Unexpected error:', data.message || 'Unknown error');
            }
        } else if (error.request) {
            // Network error
            console.error('Network error:', error.message);
        } else {
            // Other error
            console.error('Error:', error.message);
        }

        return Promise.reject(error);
    }
);

// Retry logic for failed requests
const retryRequest = async (originalRequest, retries = 3) => {
    for (let i = 0; i < retries; i++) {
        try {
            return await api(originalRequest);
        } catch (error) {
            if (i === retries - 1) throw error;

            // Wait before retrying (exponential backoff)
            const delay = Math.pow(2, i) * 1000;
            await new Promise(resolve => setTimeout(resolve, delay));
        }
    }
};

// API endpoints
export const movieAPI = {
    // Get initial movies for rating
    getInitialMovies: async () => {
        try {
            const response = await api.get('/api/movies/initial');
            return response.data;
        } catch (error) {
            console.error('Failed to fetch initial movies:', error);
            throw error;
        }
    },

    // Get specific movie details
    getMovie: async (movieId) => {
        try {
            const response = await api.get(`/api/movies/${movieId}`);
            return response.data;
        } catch (error) {
            console.error(`Failed to fetch movie ${movieId}:`, error);
            throw error;
        }
    },

    // Get recommendations based on user ratings
    getRecommendations: async (ratings, algorithm = 'hybrid') => {
        try {
            const response = await api.post('/api/recommendations', {
                ratings,
                algorithm
            });
            return response.data;
        } catch (error) {
            console.error('Failed to get recommendations:', error);
            throw error;
        }
    },

    // Health check
    healthCheck: async () => {
        try {
            const response = await api.get('/api/health');
            return response.data;
        } catch (error) {
            console.error('Health check failed:', error);
            throw error;
        }
    }
};

export default api;