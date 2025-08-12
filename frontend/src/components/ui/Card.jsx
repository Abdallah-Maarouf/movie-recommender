import { motion } from 'framer-motion';
import { forwardRef } from 'react';

/**
 * Netflix-style Card component with hover animations
 */
const Card = forwardRef(({
    children,
    className = '',
    hover = true,
    onClick,
    ...props
}, ref) => {
    const baseClasses = 'bg-netflix-darkGray rounded-lg shadow-lg overflow-hidden transition-all duration-300';
    const hoverClasses = hover ? 'hover:shadow-xl cursor-pointer' : '';
    const cardClasses = `${baseClasses} ${hoverClasses} ${className}`;

    const cardVariants = {
        initial: { scale: 1, y: 0 },
        hover: { scale: 1.02, y: -5 },
    };

    if (onClick) {
        return (
            <motion.div
                ref={ref}
                className={cardClasses}
                onClick={onClick}
                variants={cardVariants}
                initial="initial"
                whileHover={hover ? "hover" : "initial"}
                {...props}
            >
                {children}
            </motion.div>
        );
    }

    return (
        <motion.div
            ref={ref}
            className={cardClasses}
            variants={cardVariants}
            initial="initial"
            whileHover={hover ? "hover" : "initial"}
            {...props}
        >
            {children}
        </motion.div>
    );
});

Card.displayName = 'Card';

export default Card;