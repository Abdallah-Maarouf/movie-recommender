import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { movieAPI } from './services/api';
import { ERROR_MESSAGES } from './utils/constants';
import { LoadingOverlay } from './components/ui/LoadingSpinner';
import LandingPage from './pages/LandingPage';
import './index.css';

function App() {
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState(null);
    const [currentPage, setCurrentPage] = useState('landing');

    useEffect(() => {
        // Check API health and initialize app
        const initializeApp = async () => {
            try {
                setIsLoading(true);
                setError(null);

                // Check if backend is available (optional for landing page)
                try {
                    await movieAPI.healthCheck();
                } catch (apiError) {
                    console.warn('Backend not available, continuing with frontend-only mode');
                }

                // Add a small delay for better UX
                setTimeout(() => {
                    setIsLoading(false);
                }, 1000);

            } catch (err) {
                console.error('Failed to initialize app:', err);
                setError(ERROR_MESSAGES.generic);
                setIsLoading(false);
            }
        };

        initializeApp();
    }, []);

    const handleRetry = () => {
        window.location.reload();
    };

    const handleStartRating = () => {
        setCurrentPage('rating');
    };

    const handleNavigate = (page) => {
        setCurrentPage(page);
    };

    if (isLoading) {
        return (
            <LoadingOverlay message="Initializing Movie Recommender..." />
        );
    }

    if (error) {
        return (
            <div className="min-h-screen bg-netflix-black flex items-center justify-center p-4">
                <motion.div
                    className="text-center max-w-md"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                >
                    <div className="text-6xl mb-4">🎬</div>
                    <h1 className="text-2xl font-bold text-netflix-white mb-4">
                        Oops! Something went wrong
                    </h1>
                    <p className="text-netflix-lightGray mb-6">
                        {error}
                    </p>
                    <button
                        onClick={handleRetry}
                        className="btn-primary"
                    >
                        Try Again
                    </button>
                </motion.div>
            </div>
        );
    }

    return (
        <div className="min-h-screen bg-netflix-black">
            <AnimatePresence mode="wait">
                {currentPage === 'landing' && (
                    <motion.div
                        key="landing"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        transition={{ duration: 0.5 }}
                    >
                        <LandingPage
                            onStartRating={handleStartRating}
                        />
                    </motion.div>
                )}

                {currentPage === 'rating' && (
                    <motion.div
                        key="rating"
                        initial={{ opacity: 0, x: 100 }}
                        animate={{ opacity: 1, x: 0 }}
                        exit={{ opacity: 0, x: -100 }}
                        transition={{ duration: 0.5 }}
                        className="min-h-screen flex items-center justify-center"
                    >
                        <div className="text-center">
                            <h2 className="text-4xl font-bold text-netflix-white mb-4">
                                Rating Page Coming Soon
                            </h2>
                            <p className="text-netflix-lightGray mb-6">
                                This will be implemented in the next task.
                            </p>
                            <button
                                onClick={() => handleNavigate('landing')}
                                className="btn-secondary"
                            >
                                Back to Home
                            </button>
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
}

export default App;