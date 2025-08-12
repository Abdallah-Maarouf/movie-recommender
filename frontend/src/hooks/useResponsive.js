import { useState, useEffect } from 'react';
import { BREAKPOINTS, GRID_COLUMNS } from '../utils/constants';
import { getGridColumns } from '../utils/helpers';

/**
 * Hook for responsive design utilities
 * @returns {Object} Responsive utilities and screen information
 */
export const useResponsive = () => {
    const [screenSize, setScreenSize] = useState({
        width: typeof window !== 'undefined' ? window.innerWidth : 1024,
        height: typeof window !== 'undefined' ? window.innerHeight : 768,
    });

    const [isMobile, setIsMobile] = useState(false);
    const [isTablet, setIsTablet] = useState(false);
    const [isDesktop, setIsDesktop] = useState(true);

    useEffect(() => {
        const handleResize = () => {
            const width = window.innerWidth;
            const height = window.innerHeight;

            setScreenSize({ width, height });
            setIsMobile(width < BREAKPOINTS.md);
            setIsTablet(width >= BREAKPOINTS.md && width < BREAKPOINTS.lg);
            setIsDesktop(width >= BREAKPOINTS.lg);
        };

        // Set initial values
        handleResize();

        // Add event listener
        window.addEventListener('resize', handleResize);

        // Cleanup
        return () => window.removeEventListener('resize', handleResize);
    }, []);

    const gridColumns = getGridColumns(screenSize.width);

    return {
        screenSize,
        isMobile,
        isTablet,
        isDesktop,
        gridColumns,
        breakpoints: BREAKPOINTS,
    };
};

/**
 * Hook for media queries
 * @param {string} query - Media query string
 * @returns {boolean} Whether the media query matches
 */
export const useMediaQuery = (query) => {
    const [matches, setMatches] = useState(false);

    useEffect(() => {
        if (typeof window === 'undefined') return;

        const mediaQuery = window.matchMedia(query);
        setMatches(mediaQuery.matches);

        const handler = (event) => setMatches(event.matches);
        mediaQuery.addEventListener('change', handler);

        return () => mediaQuery.removeEventListener('change', handler);
    }, [query]);

    return matches;
};

/**
 * Hook for detecting touch devices
 * @returns {boolean} Whether the device supports touch
 */
export const useTouch = () => {
    const [isTouch, setIsTouch] = useState(false);

    useEffect(() => {
        const checkTouch = () => {
            setIsTouch('ontouchstart' in window || navigator.maxTouchPoints > 0);
        };

        checkTouch();
        window.addEventListener('touchstart', checkTouch, { once: true });

        return () => window.removeEventListener('touchstart', checkTouch);
    }, []);

    return isTouch;
};

/**
 * Hook for viewport dimensions
 * @returns {Object} Viewport width and height
 */
export const useViewport = () => {
    const [viewport, setViewport] = useState({
        width: typeof window !== 'undefined' ? window.innerWidth : 1024,
        height: typeof window !== 'undefined' ? window.innerHeight : 768,
    });

    useEffect(() => {
        const handleResize = () => {
            setViewport({
                width: window.innerWidth,
                height: window.innerHeight,
            });
        };

        window.addEventListener('resize', handleResize);
        return () => window.removeEventListener('resize', handleResize);
    }, []);

    return viewport;
};

export default useResponsive;