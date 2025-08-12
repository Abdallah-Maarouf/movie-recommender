/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            colors: {
                netflix: {
                    black: '#141414',
                    darkGray: '#181818',
                    gray: '#333333',
                    lightGray: '#B3B3B3',
                    red: '#E50914',
                    white: '#FFFFFF',
                    darkRed: '#B81D24',
                    accent: '#F5F5F1'
                }
            },
            fontFamily: {
                'netflix': ['Helvetica Neue', 'Helvetica', 'Arial', 'sans-serif'],
            },
            animation: {
                'fade-in': 'fadeIn 0.6s ease-in-out',
                'slide-up': 'slideUp 0.8s ease-out',
                'scale-in': 'scaleIn 0.4s ease-out',
                'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
            },
            keyframes: {
                fadeIn: {
                    '0%': { opacity: '0' },
                    '100%': { opacity: '1' },
                },
                slideUp: {
                    '0%': { transform: 'translateY(30px)', opacity: '0' },
                    '100%': { transform: 'translateY(0)', opacity: '1' },
                },
                scaleIn: {
                    '0%': { transform: 'scale(0.95)', opacity: '0' },
                    '100%': { transform: 'scale(1)', opacity: '1' },
                },
            },
            screens: {
                'xs': '475px',
            },
            spacing: {
                '18': '4.5rem',
                '88': '22rem',
            },
            backdropBlur: {
                xs: '2px',
            },
        },
    },
    plugins: [],
}