import { motion } from 'framer-motion';
import { APP_CONFIG } from '../../utils/constants';

/**
 * App Header Component - Minimal navigation header
 * Features responsive design and smooth animations
 */
const AppHeader = ({ currentPage = 'landing', onNavigate }) => {
    const headerVariants = {
        hidden: { y: -20, opacity: 0 },
        visible: {
            y: 0,
            opacity: 1,
            transition: {
                duration: 0.6,
                ease: "easeOut"
            }
        }
    };

    const logoVariants = {
        initial: { scale: 1 },
        hover: {
            scale: 1.05,
            transition: { type: "spring", stiffness: 300 }
        }
    };

    // Navigation items based on current page
    const getNavigationItems = () => {
        switch (currentPage) {
            case 'rating':
                return [
                    { label: 'Home', action: () => onNavigate?.('landing') }
                ];
            case 'recommendations':
                return [
                    { label: 'Home', action: () => onNavigate?.('landing') },
                    { label: 'Rate More', action: () => onNavigate?.('rating') }
                ];
            default:
                return [];
        }
    };

    const navigationItems = getNavigationItems();

    return (
        <motion.header
            className="bg-netflix-black/95 backdrop-blur-netflix border-b border-netflix-gray/30 sticky top-0 z-50"
            variants={headerVariants}
            initial="hidden"
            animate="visible"
        >
            <div className="container mx-auto px-4">
                <div className="flex items-center justify-between h-16">
                    {/* Logo/Brand */}
                    <motion.div
                        className="flex items-center space-x-3"
                        variants={logoVariants}
                        initial="initial"
                        whileHover="hover"
                    >
                        <div className="text-2xl">🎬</div>
                        <h1 className="text-xl md:text-2xl font-bold text-netflix-red">
                            {APP_CONFIG.name}
                        </h1>
                    </motion.div>

                    {/* Navigation */}
                    {navigationItems.length > 0 && (
                        <nav className="hidden md:flex items-center space-x-6">
                            {navigationItems.map((item, index) => (
                                <motion.button
                                    key={index}
                                    onClick={item.action}
                                    className="text-netflix-lightGray hover:text-netflix-white transition-colors duration-300 font-medium"
                                    whileHover={{ scale: 1.05 }}
                                    whileTap={{ scale: 0.95 }}
                                >
                                    {item.label}
                                </motion.button>
                            ))}
                        </nav>
                    )}

                    {/* Mobile menu button */}
                    {navigationItems.length > 0 && (
                        <div className="md:hidden">
                            <motion.button
                                className="text-netflix-lightGray hover:text-netflix-white p-2"
                                whileTap={{ scale: 0.95 }}
                                aria-label="Open menu"
                            >
                                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
                                </svg>
                            </motion.button>
                        </div>
                    )}
                </div>

                {/* Progress indicator for rating/recommendation pages */}
                {(currentPage === 'rating' || currentPage === 'recommendations') && (
                    <motion.div
                        className="pb-2"
                        initial={{ opacity: 0, scaleX: 0 }}
                        animate={{ opacity: 1, scaleX: 1 }}
                        transition={{ duration: 0.8, delay: 0.3 }}
                    >
                        <div className="flex items-center justify-center space-x-2 text-sm text-netflix-lightGray">
                            <div className={`w-3 h-3 rounded-full ${currentPage === 'rating' ? 'bg-netflix-red' : 'bg-netflix-gray'}`} />
                            <span className={currentPage === 'rating' ? 'text-netflix-white' : ''}>
                                Rate Movies
                            </span>
                            <div className="w-8 h-px bg-netflix-gray" />
                            <div className={`w-3 h-3 rounded-full ${currentPage === 'recommendations' ? 'bg-netflix-red' : 'bg-netflix-gray'}`} />
                            <span className={currentPage === 'recommendations' ? 'text-netflix-white' : ''}>
                                Get Recommendations
                            </span>
                        </div>
                    </motion.div>
                )}
            </div>
        </motion.header>
    );
};

export default AppHeader;