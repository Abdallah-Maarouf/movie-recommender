import { motion } from 'framer-motion';

/**
 * Typography components with Netflix styling
 */

export const Heading1 = ({ children, className = '', ...props }) => (
    <motion.h1
        className={`text-4xl md:text-5xl lg:text-6xl font-bold text-netflix-white text-shadow ${className}`}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        {...props}
    >
        {children}
    </motion.h1>
);

export const Heading2 = ({ children, className = '', ...props }) => (
    <motion.h2
        className={`text-2xl md:text-3xl lg:text-4xl font-bold text-netflix-white ${className}`}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        {...props}
    >
        {children}
    </motion.h2>
);

export const Heading3 = ({ children, className = '', ...props }) => (
    <motion.h3
        className={`text-xl md:text-2xl font-semibold text-netflix-white ${className}`}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        {...props}
    >
        {children}
    </motion.h3>
);

export const Subtitle = ({ children, className = '', ...props }) => (
    <motion.p
        className={`text-lg md:text-xl text-netflix-lightGray ${className}`}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        {...props}
    >
        {children}
    </motion.p>
);

export const Body = ({ children, className = '', ...props }) => (
    <p className={`text-base text-netflix-lightGray ${className}`} {...props}>
        {children}
    </p>
);

export const Caption = ({ children, className = '', ...props }) => (
    <p className={`text-sm text-netflix-lightGray ${className}`} {...props}>
        {children}
    </p>
);

export const Label = ({ children, className = '', ...props }) => (
    <label className={`text-sm font-medium text-netflix-white ${className}`} {...props}>
        {children}
    </label>
);