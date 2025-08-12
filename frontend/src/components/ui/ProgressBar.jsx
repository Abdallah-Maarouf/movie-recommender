import { motion } from 'framer-motion';
import { calculateProgress } from '../../utils/helpers';

/**
 * Netflix-style Progress Bar component
 */
const ProgressBar = ({
    current = 0,
    total = 100,
    showLabel = true,
    showPercentage = true,
    className = '',
    color = 'netflix-red',
    size = 'medium',
    animated = true,
}) => {
    const percentage = calculateProgress(current, total);

    const sizes = {
        small: 'h-2',
        medium: 'h-3',
        large: 'h-4',
    };

    const colors = {
        'netflix-red': 'bg-netflix-red',
        'netflix-white': 'bg-netflix-white',
        'netflix-gray': 'bg-netflix-gray',
    };

    const progressVariants = {
        initial: { width: 0 },
        animate: { width: `${percentage}%` },
    };

    return (
        <div className={`w-full ${className}`}>
            {showLabel && (
                <div className="flex justify-between items-center mb-2">
                    <span className="text-sm text-netflix-lightGray">
                        Progress: {current} / {total}
                    </span>
                    {showPercentage && (
                        <span className="text-sm text-netflix-white font-semibold">
                            {percentage}%
                        </span>
                    )}
                </div>
            )}

            <div className={`w-full bg-netflix-gray rounded-full ${sizes[size]} overflow-hidden`}>
                <motion.div
                    className={`${sizes[size]} ${colors[color]} rounded-full`}
                    variants={progressVariants}
                    initial={animated ? "initial" : false}
                    animate="animate"
                    transition={{ duration: 0.8, ease: "easeOut" }}
                />
            </div>
        </div>
    );
};

/**
 * Circular Progress component
 */
export const CircularProgress = ({
    current = 0,
    total = 100,
    size = 80,
    strokeWidth = 8,
    color = '#E50914',
    backgroundColor = '#333333',
    showPercentage = true,
    className = '',
}) => {
    const percentage = calculateProgress(current, total);
    const radius = (size - strokeWidth) / 2;
    const circumference = radius * 2 * Math.PI;
    const strokeDasharray = circumference;
    const strokeDashoffset = circumference - (percentage / 100) * circumference;

    return (
        <div className={`relative inline-flex items-center justify-center ${className}`}>
            <svg
                width={size}
                height={size}
                className="transform -rotate-90"
            >
                {/* Background circle */}
                <circle
                    cx={size / 2}
                    cy={size / 2}
                    r={radius}
                    stroke={backgroundColor}
                    strokeWidth={strokeWidth}
                    fill="transparent"
                />

                {/* Progress circle */}
                <motion.circle
                    cx={size / 2}
                    cy={size / 2}
                    r={radius}
                    stroke={color}
                    strokeWidth={strokeWidth}
                    fill="transparent"
                    strokeLinecap="round"
                    strokeDasharray={strokeDasharray}
                    initial={{ strokeDashoffset: circumference }}
                    animate={{ strokeDashoffset }}
                    transition={{ duration: 1, ease: "easeOut" }}
                />
            </svg>

            {showPercentage && (
                <div className="absolute inset-0 flex items-center justify-center">
                    <span className="text-netflix-white font-semibold">
                        {percentage}%
                    </span>
                </div>
            )}
        </div>
    );
};

export default ProgressBar;