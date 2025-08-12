import { motion } from 'framer-motion';

/**
 * Netflix-style Loading Spinner component
 */
const LoadingSpinner = ({
    size = 'medium',
    color = 'netflix-red',
    message = '',
    className = ''
}) => {
    const sizes = {
        small: 'h-4 w-4',
        medium: 'h-8 w-8',
        large: 'h-12 w-12',
        xlarge: 'h-16 w-16',
    };

    const colors = {
        'netflix-red': 'border-netflix-red',
        'netflix-white': 'border-netflix-white',
        'netflix-gray': 'border-netflix-gray',
    };

    const spinnerVariants = {
        animate: {
            rotate: 360,
            transition: {
                duration: 1,
                repeat: Infinity,
                ease: "linear"
            }
        }
    };

    return (
        <div className={`flex flex-col items-center justify-center ${className}`}>
            <motion.div
                className={`${sizes[size]} ${colors[color]} border-2 border-t-transparent rounded-full`}
                variants={spinnerVariants}
                animate="animate"
            />
            {message && (
                <motion.p
                    className="mt-4 text-netflix-lightGray text-center"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: 0.5 }}
                >
                    {message}
                </motion.p>
            )}
        </div>
    );
};

/**
 * Full screen loading overlay
 */
export const LoadingOverlay = ({ message = 'Loading...' }) => {
    return (
        <motion.div
            className="fixed inset-0 bg-netflix-black bg-opacity-80 flex items-center justify-center z-50"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
        >
            <LoadingSpinner size="large" message={message} />
        </motion.div>
    );
};

/**
 * Skeleton loader for content placeholders
 */
export const SkeletonLoader = ({ className = '', rows = 3 }) => {
    return (
        <div className={`animate-pulse ${className}`}>
            {Array.from({ length: rows }).map((_, index) => (
                <div
                    key={index}
                    className="bg-netflix-gray rounded h-4 mb-2"
                    style={{ width: `${Math.random() * 40 + 60}%` }}
                />
            ))}
        </div>
    );
};

export default LoadingSpinner;