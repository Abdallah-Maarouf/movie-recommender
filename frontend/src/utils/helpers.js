import { GENRE_COLORS } from './constants';

/**
 * Format movie title for display
 * @param {string} title - Movie title
 * @param {number} year - Release year
 * @returns {string} Formatted title
 */
export const formatMovieTitle = (title, year) => {
    if (!title) return 'Unknown Movie';
    return year ? `${title} (${year})` : title;
};

/**
 * Get color for a genre
 * @param {string} genre - Genre name
 * @returns {string} Color code
 */
export const getGenreColor = (genre) => {
    return GENRE_COLORS[genre] || '#B3B3B3';
};

/**
 * Format rating for display
 * @param {number} rating - Rating value
 * @param {number} decimals - Number of decimal places
 * @returns {string} Formatted rating
 */
export const formatRating = (rating, decimals = 1) => {
    if (typeof rating !== 'number' || isNaN(rating)) return 'N/A';
    return rating.toFixed(decimals);
};

/**
 * Truncate text to specified length
 * @param {string} text - Text to truncate
 * @param {number} maxLength - Maximum length
 * @returns {string} Truncated text
 */
export const truncateText = (text, maxLength = 100) => {
    if (!text || text.length <= maxLength) return text;
    return text.substring(0, maxLength).trim() + '...';
};

/**
 * Debounce function to limit API calls
 * @param {Function} func - Function to debounce
 * @param {number} wait - Wait time in milliseconds
 * @returns {Function} Debounced function
 */
export const debounce = (func, wait) => {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
};

/**
 * Throttle function to limit function calls
 * @param {Function} func - Function to throttle
 * @param {number} limit - Time limit in milliseconds
 * @returns {Function} Throttled function
 */
export const throttle = (func, limit) => {
    let inThrottle;
    return function executedFunction(...args) {
        if (!inThrottle) {
            func.apply(this, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    };
};

/**
 * Check if device supports touch
 * @returns {boolean} True if touch is supported
 */
export const isTouchDevice = () => {
    return 'ontouchstart' in window || navigator.maxTouchPoints > 0;
};

/**
 * Get responsive grid columns based on screen width
 * @param {number} width - Screen width
 * @returns {number} Number of columns
 */
export const getGridColumns = (width) => {
    if (width < 640) return 1; // mobile
    if (width < 768) return 2; // tablet
    if (width < 1024) return 3; // desktop
    return 4; // large desktop
};

/**
 * Generate a random ID
 * @returns {string} Random ID
 */
export const generateId = () => {
    return Math.random().toString(36).substring(2) + Date.now().toString(36);
};

/**
 * Validate rating value
 * @param {number} rating - Rating to validate
 * @returns {boolean} True if valid
 */
export const isValidRating = (rating) => {
    return typeof rating === 'number' && rating >= 1 && rating <= 5;
};

/**
 * Calculate progress percentage
 * @param {number} current - Current value
 * @param {number} total - Total value
 * @returns {number} Percentage (0-100)
 */
export const calculateProgress = (current, total) => {
    if (total === 0) return 0;
    return Math.min(Math.round((current / total) * 100), 100);
};

/**
 * Format large numbers for display
 * @param {number} num - Number to format
 * @returns {string} Formatted number
 */
export const formatNumber = (num) => {
    if (num >= 1000000) {
        return (num / 1000000).toFixed(1) + 'M';
    }
    if (num >= 1000) {
        return (num / 1000).toFixed(1) + 'K';
    }
    return num.toString();
};

/**
 * Get image URL with fallback
 * @param {string} url - Image URL
 * @param {string} fallback - Fallback URL
 * @returns {string} Image URL
 */
export const getImageUrl = (url, fallback = '/placeholder-movie.jpg') => {
    return url || fallback;
};

/**
 * Shuffle array using Fisher-Yates algorithm
 * @param {Array} array - Array to shuffle
 * @returns {Array} Shuffled array
 */
export const shuffleArray = (array) => {
    const shuffled = [...array];
    for (let i = shuffled.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
    }
    return shuffled;
};

/**
 * Deep clone an object
 * @param {Object} obj - Object to clone
 * @returns {Object} Cloned object
 */
export const deepClone = (obj) => {
    if (obj === null || typeof obj !== 'object') return obj;
    if (obj instanceof Date) return new Date(obj.getTime());
    if (obj instanceof Array) return obj.map(item => deepClone(item));
    if (typeof obj === 'object') {
        const clonedObj = {};
        for (const key in obj) {
            if (obj.hasOwnProperty(key)) {
                clonedObj[key] = deepClone(obj[key]);
            }
        }
        return clonedObj;
    }
};

/**
 * Check if object is empty
 * @param {Object} obj - Object to check
 * @returns {boolean} True if empty
 */
export const isEmpty = (obj) => {
    if (obj == null) return true;
    if (Array.isArray(obj) || typeof obj === 'string') return obj.length === 0;
    return Object.keys(obj).length === 0;
};

/**
 * Create a delay promise
 * @param {number} ms - Milliseconds to delay
 * @returns {Promise} Promise that resolves after delay
 */
export const delay = (ms) => {
    return new Promise(resolve => setTimeout(resolve, ms));
};