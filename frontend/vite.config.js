import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
    plugins: [react()],
    server: {
        port: 3000,
        host: true,
    },
    build: {
        outDir: 'dist',
        sourcemap: true,
        rollupOptions: {
            output: {
                manualChunks: {
                    vendor: ['react', 'react-dom'],
                    motion: ['framer-motion'],
                    utils: ['axios'],
                },
            },
        },
    },
    optimizeDeps: {
        include: ['react', 'react-dom', 'framer-motion', 'axios'],
    },
})