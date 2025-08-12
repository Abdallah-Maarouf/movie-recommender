import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import App from '../src/App';

// Mock framer-motion to avoid animation issues in tests
vi.mock('framer-motion', () => ({
    motion: {
        div: ({ children, ...props }) => <div {...props}>{children}</div>,
        h1: ({ children, ...props }) => <h1 {...props}>{children}</h1>,
        h2: ({ children, ...props }) => <h2 {...props}>{children}</h2>,
        p: ({ children, ...props }) => <p {...props}>{children}</p>,
        button: ({ children, ...props }) => <button {...props}>{children}</button>,
        section: ({ children, ...props }) => <section {...props}>{children}</section>,
        header: ({ children, ...props }) => <header {...props}>{children}</header>,
        span: ({ children, ...props }) => <span {...props}>{children}</span>,
    },
    AnimatePresence: ({ children }) => children,
}));

// Mock API service
vi.mock('../src/services/api', () => ({
    movieAPI: {
        healthCheck: vi.fn().mockResolvedValue({ status: 'ok' }),
    },
}));

describe('App Component', () => {
    it('renders the landing page after loading', async () => {
        render(<App />);

        // Wait for loading to complete and landing page to appear
        await waitFor(() => {
            expect(screen.getByText('Movie Recommender')).toBeInTheDocument();
        });

        expect(screen.getByText('Discover Movies')).toBeInTheDocument();
        expect(screen.getByText("You'll Love")).toBeInTheDocument();
    });

    it('navigates to rating page when CTA is clicked', async () => {
        render(<App />);

        // Wait for landing page to load
        await waitFor(() => {
            expect(screen.getByText('Start Rating Movies')).toBeInTheDocument();
        });

        // Click the CTA button
        const ctaButton = screen.getByText('Start Rating Movies');
        fireEvent.click(ctaButton);

        // Should navigate to rating page (placeholder)
        await waitFor(() => {
            expect(screen.getByText('Rating Page Coming Soon')).toBeInTheDocument();
        });
    });

    it('handles API errors gracefully', async () => {
        // Mock API to throw error
        const mockHealthCheck = vi.fn().mockRejectedValue(new Error('Network error'));
        vi.doMock('../src/services/api', () => ({
            movieAPI: {
                healthCheck: mockHealthCheck,
            },
        }));

        render(<App />);

        // Should still render the landing page even if API fails
        await waitFor(() => {
            expect(screen.getByText('Movie Recommender')).toBeInTheDocument();
        });
    });
});