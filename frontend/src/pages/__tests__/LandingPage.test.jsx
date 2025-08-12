import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import LandingPage from '../LandingPage';

// Mock framer-motion to avoid animation issues in tests
vi.mock('framer-motion', () => ({
    motion: {
        div: ({ children, ...props }) => <div {...props}>{children}</div>,
        h1: ({ children, ...props }) => <h1 {...props}>{children}</h1>,
        p: ({ children, ...props }) => <p {...props}>{children}</p>,
        button: ({ children, ...props }) => <button {...props}>{children}</button>,
        section: ({ children, ...props }) => <section {...props}>{children}</section>,
    },
    AnimatePresence: ({ children }) => children,
}));

describe('LandingPage', () => {
    const mockOnStartRating = vi.fn();

    beforeEach(() => {
        mockOnStartRating.mockClear();
    });

    it('renders the landing page with main elements', () => {
        render(<LandingPage onStartRating={mockOnStartRating} />);

        // Check for main headline
        expect(screen.getByText('Discover Movies')).toBeInTheDocument();
        expect(screen.getByText("You'll Love")).toBeInTheDocument();

        // Check for subtitle
        expect(screen.getByText(/Get personalized movie recommendations/)).toBeInTheDocument();

        // Check for CTA button
        expect(screen.getByText('Start Rating Movies')).toBeInTheDocument();
    });

    it('displays feature highlights', () => {
        render(<LandingPage onStartRating={mockOnStartRating} />);

        expect(screen.getByText('AI-Powered Recommendations')).toBeInTheDocument();
        expect(screen.getByText('Personalized for You')).toBeInTheDocument();
        expect(screen.getByText('No Sign-up Required')).toBeInTheDocument();
    });

    it('shows the "How It Works" section', () => {
        render(<LandingPage onStartRating={mockOnStartRating} />);

        expect(screen.getByText('How It Works')).toBeInTheDocument();
        expect(screen.getByText('Rate Movies')).toBeInTheDocument();
        expect(screen.getByText('AI Analysis')).toBeInTheDocument();
        expect(screen.getByText('Get Recommendations')).toBeInTheDocument();
    });

    it('displays feature benefits', () => {
        render(<LandingPage onStartRating={mockOnStartRating} />);

        expect(screen.getByText('Privacy First')).toBeInTheDocument();
        expect(screen.getByText('Instant Results')).toBeInTheDocument();
        expect(screen.getByText('Smart Explanations')).toBeInTheDocument();
        expect(screen.getByText('Mobile Friendly')).toBeInTheDocument();
    });

    it('calls onStartRating when CTA button is clicked', async () => {
        render(<LandingPage onStartRating={mockOnStartRating} />);

        const ctaButton = screen.getByText('Start Rating Movies');
        fireEvent.click(ctaButton);

        await waitFor(() => {
            expect(mockOnStartRating).toHaveBeenCalledTimes(1);
        });
    });

    it('shows loading state when CTA is clicked', async () => {
        render(<LandingPage onStartRating={mockOnStartRating} />);

        const ctaButton = screen.getByText('Start Rating Movies');
        fireEvent.click(ctaButton);

        // Should show loading text
        expect(screen.getByText('Preparing Movies...')).toBeInTheDocument();
    });

    it('includes footer with social links', () => {
        render(<LandingPage onStartRating={mockOnStartRating} />);

        expect(screen.getByText(/Built with React, FastAPI, and Machine Learning/)).toBeInTheDocument();

        // Check for social links
        const githubLink = screen.getByLabelText('View source code on GitHub');
        const linkedinLink = screen.getByLabelText('Connect on LinkedIn');

        expect(githubLink).toBeInTheDocument();
        expect(linkedinLink).toBeInTheDocument();
    });

    it('has proper accessibility attributes', () => {
        render(<LandingPage onStartRating={mockOnStartRating} />);

        // Check for proper heading hierarchy
        const mainHeading = screen.getByRole('heading', { level: 1 });
        expect(mainHeading).toBeInTheDocument();

        // Check for accessible button
        const ctaButton = screen.getByRole('button', { name: /Start Rating Movies/ });
        expect(ctaButton).toBeInTheDocument();
    });

    it('displays technology tags', () => {
        render(<LandingPage onStartRating={mockOnStartRating} />);

        expect(screen.getByText('Machine Learning')).toBeInTheDocument();
        expect(screen.getByText('Collaborative Filtering')).toBeInTheDocument();
        expect(screen.getByText('Content-Based Filtering')).toBeInTheDocument();
        expect(screen.getByText('React & FastAPI')).toBeInTheDocument();
        expect(screen.getByText('MovieLens Dataset')).toBeInTheDocument();
    });
});