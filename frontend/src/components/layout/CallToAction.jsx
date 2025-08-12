import { motion } from 'framer-motion';
import Button from '../ui/Button';

/**
 * Call to Action Component - Prominent, animated CTA button
 * Features hover effects, loading states, and accessibility
 */
const CallToAction = ({
    onClick,
    isLoading = false,
    text = "Get Started",
    loadingText = "Loading...",
    variant = "primary",
    size = "large",
    className = ""
}) => {
    const buttonVariants = {
        initial: { scale: 1 },
        hover: {
            scale: 1.05,
            boxShadow: "0 10px 30px rgba(229, 9, 20, 0.4)"
        },
        tap: { scale: 0.95 }
    };

    const glowVariants = {
        initial: { opacity: 0 },
        hover: {
            opacity: 1,
            transition: { duration: 0.3 }
        }
    };

    return (
        <div className="relative inline-block">
            {/* Glow effect */}
            <motion.div
                className="absolute inset-0 bg-netflix-red rounded-md blur-xl opacity-0"
                variants={glowVariants}
                initial="initial"
                whileHover="hover"
            />

            {/* Main button */}
            <motion.div
                variants={buttonVariants}
                initial="initial"
                whileHover={!isLoading ? "hover" : "initial"}
                whileTap={!isLoading ? "tap" : "initial"}
                className="relative"
            >
                <Button
                    onClick={onClick}
                    variant={variant}
                    size={size}
                    loading={isLoading}
                    disabled={isLoading}
                    className={`
                        relative z-10 
                        text-xl font-bold 
                        py-4 px-12 
                        shadow-lg 
                        transition-all duration-300
                        ${className}
                    `}
                    aria-label={isLoading ? loadingText : text}
                >
                    {isLoading ? (
                        <div className="flex items-center justify-center">
                            <motion.div
                                className="w-6 h-6 border-2 border-white border-t-transparent rounded-full mr-3"
                                animate={{ rotate: 360 }}
                                transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                            />
                            {loadingText}
                        </div>
                    ) : (
                        <div className="flex items-center justify-center">
                            <span className="mr-2">{text}</span>
                            <motion.span
                                animate={{ x: [0, 5, 0] }}
                                transition={{ duration: 1.5, repeat: Infinity }}
                            >
                                →
                            </motion.span>
                        </div>
                    )}
                </Button>
            </motion.div>

            {/* Pulse effect when loading */}
            {isLoading && (
                <motion.div
                    className="absolute inset-0 bg-netflix-red rounded-md opacity-20"
                    animate={{ scale: [1, 1.1, 1] }}
                    transition={{ duration: 1.5, repeat: Infinity }}
                />
            )}
        </div>
    );
};

export default CallToAction;