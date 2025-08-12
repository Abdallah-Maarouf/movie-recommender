import { motion } from 'framer-motion';
import { forwardRef } from 'react';

/**
 * Netflix-style Button component with animations
 */
const Button = forwardRef(({
    children,
    variant = 'primary',
    size = 'medium',
    disabled = false,
    loading = false,
    onClick,
    className = '',
    type = 'button',
    ...props
}, ref) => {
    const baseClasses = 'font-semibold rounded-md transition-all duration-300 focus:outline-none focus:ring-2 focus:ring-opacity-50 disabled:opacity-50 disabled:cursor-not-allowed';

    const variants = {
        primary: 'bg-netflix-red hover:bg-netflix-darkRed text-white focus:ring-netflix-red',
        secondary: 'bg-netflix-gray hover:bg-netflix-lightGray hover:text-netflix-black text-white focus:ring-netflix-gray',
        outline: 'border-2 border-netflix-red text-netflix-red hover:bg-netflix-red hover:text-white focus:ring-netflix-red',
        ghost: 'text-netflix-lightGray hover:text-white hover:bg-netflix-gray focus:ring-netflix-gray',
    };

    const sizes = {
        small: 'py-2 px-4 text-sm',
        medium: 'py-3 px-6 text-base',
        large: 'py-4 px-8 text-lg',
    };

    const buttonClasses = `${baseClasses} ${variants[variant]} ${sizes[size]} ${className}`;

    const buttonVariants = {
        initial: { scale: 1 },
        hover: { scale: 1.05 },
        tap: { scale: 0.95 },
    };

    return (
        <motion.button
            ref={ref}
            type={type}
            className={buttonClasses}
            disabled={disabled || loading}
            onClick={onClick}
            variants={buttonVariants}
            initial="initial"
            whileHover={!disabled ? "hover" : "initial"}
            whileTap={!disabled ? "tap" : "initial"}
            {...props}
        >
            {loading ? (
                <div className="flex items-center justify-center">
                    <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-2"></div>
                    Loading...
                </div>
            ) : (
                children
            )}
        </motion.button>
    );
});

Button.displayName = 'Button';

export default Button;