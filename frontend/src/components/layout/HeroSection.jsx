import { motion } from 'framer-motion';
import CallToAction from './CallToAction';

/**
 * Hero Section Component - Full-screen hero with Netflix-inspired design
 * Features gradient background, compelling copy, and prominent CTA
 */
const HeroSection = ({ onStartRating, isLoading }) => {
    const containerVariants = {
        hidden: { opacity: 0 },
        visible: {
            opacity: 1,
            transition: {
                duration: 0.8,
                staggerChildren: 0.3
            }
        }
    };

    const itemVariants = {
        hidden: { y: 30, opacity: 0 },
        visible: {
            y: 0,
            opacity: 1,
            transition: {
                duration: 0.8,
                ease: "easeOut"
            }
        }
    };

    const backgroundVariants = {
        hidden: { scale: 1.1, opacity: 0 },
        visible: {
            scale: 1,
            opacity: 1,
            transition: {
                duration: 1.2,
                ease: "easeOut"
            }
        }
    };

    return (
        <section className="relative min-h-screen flex items-center justify-center overflow-hidden">
            {/* Background with gradient overlay */}
            <motion.div
                className="absolute inset-0 hero-gradient"
                variants={backgroundVariants}
                initial="hidden"
                animate="visible"
            />

            {/* Subtle background pattern */}
            <div className="absolute inset-0 opacity-5">
                <div className="absolute inset-0" style={{
                    backgroundImage: `url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.1'%3E%3Ccircle cx='30' cy='30' r='2'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E")`,
                    backgroundSize: '60px 60px'
                }} />
            </div>

            {/* Floating movie icons */}
            <div className="absolute inset-0 pointer-events-none">
                <motion.div
                    className="absolute top-20 left-10 text-4xl opacity-20"
                    animate={{
                        y: [0, -20, 0],
                        rotate: [0, 5, 0]
                    }}
                    transition={{
                        duration: 6,
                        repeat: Infinity,
                        ease: "easeInOut"
                    }}
                >
                    🎬
                </motion.div>
                <motion.div
                    className="absolute top-32 right-16 text-3xl opacity-20"
                    animate={{
                        y: [0, -15, 0],
                        rotate: [0, -5, 0]
                    }}
                    transition={{
                        duration: 8,
                        repeat: Infinity,
                        ease: "easeInOut",
                        delay: 1
                    }}
                >
                    🍿
                </motion.div>
                <motion.div
                    className="absolute bottom-32 left-20 text-3xl opacity-20"
                    animate={{
                        y: [0, -10, 0],
                        rotate: [0, 3, 0]
                    }}
                    transition={{
                        duration: 7,
                        repeat: Infinity,
                        ease: "easeInOut",
                        delay: 2
                    }}
                >
                    🎭
                </motion.div>
                <motion.div
                    className="absolute bottom-20 right-10 text-4xl opacity-20"
                    animate={{
                        y: [0, -25, 0],
                        rotate: [0, -3, 0]
                    }}
                    transition={{
                        duration: 9,
                        repeat: Infinity,
                        ease: "easeInOut",
                        delay: 0.5
                    }}
                >
                    🎪
                </motion.div>
            </div>

            {/* Main content */}
            <motion.div
                className="relative z-10 text-center px-4 max-w-4xl mx-auto"
                variants={containerVariants}
                initial="hidden"
                animate="visible"
            >
                {/* Main headline */}
                <motion.h1
                    className="text-5xl md:text-7xl lg:text-8xl font-bold text-netflix-white mb-6 text-shadow"
                    variants={itemVariants}
                >
                    Discover Movies
                    <br />
                    <span className="text-netflix-red">You'll Love</span>
                </motion.h1>

                {/* Subtitle */}
                <motion.p
                    className="text-xl md:text-2xl lg:text-3xl text-netflix-lightGray mb-8 max-w-3xl mx-auto leading-relaxed"
                    variants={itemVariants}
                >
                    Get personalized movie recommendations powered by advanced machine learning.
                    Rate movies you've seen and discover your next favorite film.
                </motion.p>

                {/* Feature highlights */}
                <motion.div
                    className="flex flex-wrap justify-center gap-6 mb-12 text-netflix-lightGray"
                    variants={itemVariants}
                >
                    <div className="flex items-center space-x-2">
                        <span className="text-netflix-red">✨</span>
                        <span>AI-Powered Recommendations</span>
                    </div>
                    <div className="flex items-center space-x-2">
                        <span className="text-netflix-red">🎯</span>
                        <span>Personalized for You</span>
                    </div>
                    <div className="flex items-center space-x-2">
                        <span className="text-netflix-red">🚀</span>
                        <span>No Sign-up Required</span>
                    </div>
                </motion.div>

                {/* Call to action */}
                <motion.div variants={itemVariants}>
                    <CallToAction
                        onClick={onStartRating}
                        isLoading={isLoading}
                        text="Start Rating Movies"
                        loadingText="Preparing Movies..."
                    />
                </motion.div>

                {/* Additional info */}
                <motion.p
                    className="text-sm text-netflix-lightGray mt-6 opacity-75"
                    variants={itemVariants}
                >
                    Rate 15+ movies to get your personalized recommendations
                </motion.p>
            </motion.div>

            {/* Scroll indicator */}
            <motion.div
                className="absolute bottom-8 left-1/2 transform -translate-x-1/2"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 2, duration: 0.8 }}
            >
                <motion.div
                    className="w-6 h-10 border-2 border-netflix-lightGray rounded-full flex justify-center"
                    animate={{ opacity: [0.5, 1, 0.5] }}
                    transition={{ duration: 2, repeat: Infinity }}
                >
                    <motion.div
                        className="w-1 h-3 bg-netflix-lightGray rounded-full mt-2"
                        animate={{ y: [0, 12, 0] }}
                        transition={{ duration: 2, repeat: Infinity }}
                    />
                </motion.div>
            </motion.div>
        </section>
    );
};

export default HeroSection;