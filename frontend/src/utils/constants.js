// App configuration constants
export const APP_CONFIG = {
    name: 'Movie Recommender',
    description: 'Discover movies you\'ll love with AI-powered recommendations',
    minRatingsRequired: 15,
    maxRecommendations: 20,
    ratingScale: {
        min: 1,
        max: 5
    }
};

// API endpoints
export const API_ENDPOINTS = {
    movies: {
        initial: '/api/movies/initial',
        details: '/api/movies',
    },
    recommendations: '/api/recommendations',
    health: '/api/health'
};

// UI constants
export const BREAKPOINTS = {
    xs: 475,
    sm: 640,
    md: 768,
    lg: 1024,
    xl: 1280,
    '2xl': 1536
};

export const GRID_COLUMNS = {
    mobile: 1,
    tablet: 2,
    desktop: 3,
    large: 4
};

// Animation durations (in milliseconds)
export const ANIMATION_DURATION = {
    fast: 200,
    normal: 300,
    slow: 500,
    verySlow: 800
};

// Rating interface constants
export const RATING_CONFIG = {
    stars: 5,
    labels: {
        1: 'Terrible',
        2: 'Bad',
        3: 'Okay',
        4: 'Good',
        5: 'Excellent'
    }
};

// Error messages
export const ERROR_MESSAGES = {
    network: 'Network error. Please check your connection and try again.',
    server: 'Server error. Please try again later.',
    notFound: 'The requested resource was not found.',
    validation: 'Please check your input and try again.',
    minRatings: `Please rate at least ${APP_CONFIG.minRatingsRequired} movies to get recommendations.`,
    generic: 'Something went wrong. Please try again.'
};

// Success messages
export const SUCCESS_MESSAGES = {
    ratingSaved: 'Rating saved successfully!',
    recommendationsLoaded: 'Recommendations loaded successfully!',
    dataLoaded: 'Data loaded successfully!'
};

// Loading messages
export const LOADING_MESSAGES = {
    movies: 'Loading movies...',
    recommendations: 'Generating your personalized recommendations...',
    updating: 'Updating recommendations...',
    saving: 'Saving your rating...'
};

// Local storage keys
export const STORAGE_KEYS = {
    ratings: 'movieRatings',
    preferences: 'userPreferences',
    session: 'sessionData'
};

// Netflix-style genre colors for visual variety
export const GENRE_COLORS = {
    Action: '#FF6B6B',
    Adventure: '#4ECDC4',
    Animation: '#45B7D1',
    Comedy: '#FFA07A',
    Crime: '#98D8C8',
    Documentary: '#F7DC6F',
    Drama: '#BB8FCE',
    Family: '#85C1E9',
    Fantasy: '#F8C471',
    History: '#82E0AA',
    Horror: '#EC7063',
    Music: '#F1948A',
    Mystery: '#85C1E9',
    Romance: '#F1948A',
    'Science Fiction': '#5DADE2',
    'TV Movie': '#A569BD',
    Thriller: '#EC7063',
    War: '#A569BD',
    Western: '#D7BDE2'
};