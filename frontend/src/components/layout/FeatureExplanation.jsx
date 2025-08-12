import { motion } from 'framer-motion';

/**
 * Feature Explanation Component - How the recommendation system works
 * Visual process flow with animations and engaging descriptions
 */
const FeatureExplanation = () => {
    const containerVariants = {
        hidden: { opacity: 0 },
        visible: {
            opacity: 1,
            transition: {
                duration: 0.8,
                staggerChildren: 0.2
            }
        }
    };

    const itemVariants = {
        hidden: { y: 50, opacity: 0 },
        visible: {
            y: 0,
            opacity: 1,
            transition: {
                duration: 0.8,
                ease: "easeOut"
            }
        }
    };

    const steps = [
        {
            icon: "⭐",
            title: "Rate Movies",
            description: "Rate movies you've watched to help us understand your taste",
            detail: "Our system needs at least 15 ratings to generate accurate recommendations"
        },
        {
            icon: "🤖",
            title: "AI Analysis",
            description: "Advanced machine learning algorithms analyze your preferences",
            detail: "We use collaborative filtering and content-based approaches for accuracy"
        },
        {
            icon: "🎯",
            title: "Get Recommendations",
            description: "Receive personalized movie suggestions tailored just for you",
            detail: "Each recommendation comes with an explanation of why we think you'll like it"
        }
    ];

    const features = [
        {
            icon: "🔒",
            title: "Privacy First",
            description: "No account required. Your data stays in your browser session."
        },
        {
            icon: "⚡",
            title: "Instant Results",
            description: "Get recommendations in seconds with our optimized ML models."
        },
        {
            icon: "🎨",
            title: "Smart Explanations",
            description: "Understand why each movie was recommended to you."
        },
        {
            icon: "📱",
            title: "Mobile Friendly",
            description: "Works perfectly on all devices - phone, tablet, or desktop."
        }
    ];

    return (
        <section className="py-20 bg-netflix-darkGray">
            <div className="container mx-auto px-4">
                {/* How it works section */}
                <motion.div
                    className="text-center mb-16"
                    initial={{ opacity: 0, y: 30 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true }}
                    transition={{ duration: 0.8 }}
                >
                    <h2 className="text-4xl md:text-5xl font-bold text-netflix-white mb-6">
                        How It Works
                    </h2>
                    <p className="text-xl text-netflix-lightGray max-w-2xl mx-auto">
                        Our AI-powered recommendation engine learns from your movie preferences
                        to suggest films you'll actually want to watch.
                    </p>
                </motion.div>

                {/* Process steps */}
                <motion.div
                    className="grid md:grid-cols-3 gap-8 mb-20"
                    variants={containerVariants}
                    initial="hidden"
                    whileInView="visible"
                    viewport={{ once: true }}
                >
                    {steps.map((step, index) => (
                        <motion.div
                            key={index}
                            className="text-center group"
                            variants={itemVariants}
                        >
                            <motion.div
                                className="relative mb-6"
                                whileHover={{ scale: 1.1 }}
                                transition={{ type: "spring", stiffness: 300 }}
                            >
                                <div className="w-20 h-20 mx-auto bg-netflix-red rounded-full flex items-center justify-center text-4xl mb-4 group-hover:shadow-lg group-hover:shadow-netflix-red/30 transition-all duration-300">
                                    {step.icon}
                                </div>
                                <div className="absolute -top-2 -right-2 w-8 h-8 bg-netflix-white text-netflix-black rounded-full flex items-center justify-center text-sm font-bold">
                                    {index + 1}
                                </div>
                            </motion.div>

                            <h3 className="text-2xl font-bold text-netflix-white mb-4">
                                {step.title}
                            </h3>
                            <p className="text-netflix-lightGray mb-3">
                                {step.description}
                            </p>
                            <p className="text-sm text-netflix-lightGray opacity-75">
                                {step.detail}
                            </p>
                        </motion.div>
                    ))}
                </motion.div>

                {/* Arrow indicators between steps */}
                <div className="hidden md:flex justify-center items-center mb-20 -mt-16">
                    <div className="flex items-center space-x-32">
                        <motion.div
                            className="text-netflix-red text-3xl"
                            animate={{ x: [0, 10, 0] }}
                            transition={{ duration: 2, repeat: Infinity }}
                        >
                            →
                        </motion.div>
                        <motion.div
                            className="text-netflix-red text-3xl"
                            animate={{ x: [0, 10, 0] }}
                            transition={{ duration: 2, repeat: Infinity, delay: 0.5 }}
                        >
                            →
                        </motion.div>
                    </div>
                </div>

                {/* Features grid */}
                <motion.div
                    className="text-center mb-12"
                    initial={{ opacity: 0, y: 30 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true }}
                    transition={{ duration: 0.8 }}
                >
                    <h2 className="text-3xl md:text-4xl font-bold text-netflix-white mb-6">
                        Why Choose Our Recommender?
                    </h2>
                </motion.div>

                <motion.div
                    className="grid sm:grid-cols-2 lg:grid-cols-4 gap-6"
                    variants={containerVariants}
                    initial="hidden"
                    whileInView="visible"
                    viewport={{ once: true }}
                >
                    {features.map((feature, index) => (
                        <motion.div
                            key={index}
                            className="bg-netflix-black rounded-lg p-6 text-center hover:bg-netflix-gray transition-colors duration-300"
                            variants={itemVariants}
                            whileHover={{ y: -5 }}
                        >
                            <div className="text-4xl mb-4">{feature.icon}</div>
                            <h3 className="text-lg font-semibold text-netflix-white mb-2">
                                {feature.title}
                            </h3>
                            <p className="text-netflix-lightGray text-sm">
                                {feature.description}
                            </p>
                        </motion.div>
                    ))}
                </motion.div>

                {/* Technology showcase */}
                <motion.div
                    className="mt-20 text-center"
                    initial={{ opacity: 0, y: 30 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true }}
                    transition={{ duration: 0.8 }}
                >
                    <h3 className="text-2xl font-bold text-netflix-white mb-6">
                        Powered by Advanced Technology
                    </h3>
                    <div className="flex flex-wrap justify-center gap-4 text-netflix-lightGray">
                        <span className="bg-netflix-black px-4 py-2 rounded-full text-sm">
                            Machine Learning
                        </span>
                        <span className="bg-netflix-black px-4 py-2 rounded-full text-sm">
                            Collaborative Filtering
                        </span>
                        <span className="bg-netflix-black px-4 py-2 rounded-full text-sm">
                            Content-Based Filtering
                        </span>
                        <span className="bg-netflix-black px-4 py-2 rounded-full text-sm">
                            React & FastAPI
                        </span>
                        <span className="bg-netflix-black px-4 py-2 rounded-full text-sm">
                            MovieLens Dataset
                        </span>
                    </div>
                </motion.div>
            </div>
        </section>
    );
};

export default FeatureExplanation;