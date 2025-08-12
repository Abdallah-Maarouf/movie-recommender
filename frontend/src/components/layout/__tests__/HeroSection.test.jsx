import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import HeroSection from '../HeroSection';

// Mock framer-motion
vi.mock('framer-motion', () => ({
    motion: {
        div: ({ children, ...props }) => <div {...props}>{children}</div>,
        h1: ({ children, ...props }) => <h1 {...props}>{children}</h1>,
        p: ({ children, ...props }) => <p {...props}>{children}</p>,
        section: ({ children, ...props }) => <section {...props}>{children}</section>,
    },
}));

describe('HeroSection', () => {
    const mockOnStartRating = vi.fn();

    beforeEach(() => {
        mockOnStartRating.mockClear();
    });

    it('renders the hero section with main content', () => {
        render(<HeroSection onStartRating={mockOnStartRating} />);

        expect(screen.getByText('Discover Movies')).toBeInTheDocument();
        expect(screen.getByText("You'll Love")).toBeInTheDocument();
        expect(screen.getByText(/Get personalized movie recommendations/)).toBeInTheDocument();
    });

    it('displays feature highlights with icons', () => {
        render(<HeroSection onStartRating={mockOnStartRating} />);

        expect(screen.getByText('AI-Powered Recommendations')).toBeInTheDocument();
        expect(screen.getByText('Personalized for You')).toBeInTheDocument();
        expect(screen.getByText('No Sign-up Required')).toBeInTheDocument();
    });

    it('shows the call to action button', () => {
        render(<HeroSection onStartRating={mockOnStartRating} />);

        const ctaButton = screen.getByText('Start Rating Movies');
        expect(ctaButton).toBeInTheDocument();
    });

    it('displays additional info about rating requirement', () => {
        render(<HeroSection onStartRating={mockOnStartRating} />);

        expect(screen.getByText('Rate 15+ movies to get your personalized recommendations')).toBeInTheDocument();
    });

    it('calls onStartRating when CTA is clicked', () => {
        render(<HeroSection onStartRating={mockOnStartRating} />);

        const ctaButton = screen.getByText('Start Rating Movies');
        fireEvent.click(ctaButton);

        expect(mockOnStartRating).toHaveBeenCalledTimes(1);
    });

    it('shows loading state when isLoading is true', () => {
        render(<HeroSection onStartRating={mockOnStartRating} isLoading={true} />);

        expect(screen.getByText('Preparing Movies...')).toBeInTheDocument();
    });

    it('has proper semantic structure', () => {
        render(<HeroSection onStartRating={mockOnStartRating} />);

        // Should be wrapped in a section
        const section = screen.getByRole('region');
        expect(section).toBeInTheDocument();

        // Should have main heading
        const heading = screen.getByRole('heading', { level: 1 });
        expect(heading).toBeInTheDocument();
    });
});