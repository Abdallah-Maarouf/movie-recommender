import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import AppHeader from '../AppHeader';

// Mock framer-motion
vi.mock('framer-motion', () => ({
    motion: {
        header: ({ children, ...props }) => <header {...props}>{children}</header>,
        div: ({ children, ...props }) => <div {...props}>{children}</div>,
        button: ({ children, ...props }) => <button {...props}>{children}</button>,
        h1: ({ children, ...props }) => <h1 {...props}>{children}</h1>,
    },
}));

describe('AppHeader', () => {
    const mockOnNavigate = vi.fn();

    beforeEach(() => {
        mockOnNavigate.mockClear();
    });

    it('renders the app logo and title', () => {
        render(<AppHeader />);

        expect(screen.getByText('Movie Recommender')).toBeInTheDocument();
        expect(screen.getByText('🎬')).toBeInTheDocument();
    });

    it('shows no navigation items on landing page', () => {
        render(<AppHeader currentPage="landing" onNavigate={mockOnNavigate} />);

        expect(screen.queryByText('Home')).not.toBeInTheDocument();
        expect(screen.queryByText('Rate More')).not.toBeInTheDocument();
    });

    it('shows Home navigation on rating page', () => {
        render(<AppHeader currentPage="rating" onNavigate={mockOnNavigate} />);

        expect(screen.getByText('Home')).toBeInTheDocument();
    });

    it('shows Home and Rate More navigation on recommendations page', () => {
        render(<AppHeader currentPage="recommendations" onNavigate={mockOnNavigate} />);

        expect(screen.getByText('Home')).toBeInTheDocument();
        expect(screen.getByText('Rate More')).toBeInTheDocument();
    });

    it('calls onNavigate when Home is clicked', () => {
        render(<AppHeader currentPage="rating" onNavigate={mockOnNavigate} />);

        const homeButton = screen.getByText('Home');
        fireEvent.click(homeButton);

        expect(mockOnNavigate).toHaveBeenCalledWith('landing');
    });

    it('calls onNavigate when Rate More is clicked', () => {
        render(<AppHeader currentPage="recommendations" onNavigate={mockOnNavigate} />);

        const rateMoreButton = screen.getByText('Rate More');
        fireEvent.click(rateMoreButton);

        expect(mockOnNavigate).toHaveBeenCalledWith('rating');
    });

    it('shows progress indicator on rating page', () => {
        render(<AppHeader currentPage="rating" onNavigate={mockOnNavigate} />);

        expect(screen.getByText('Rate Movies')).toBeInTheDocument();
        expect(screen.getByText('Get Recommendations')).toBeInTheDocument();
    });

    it('shows progress indicator on recommendations page', () => {
        render(<AppHeader currentPage="recommendations" onNavigate={mockOnNavigate} />);

        expect(screen.getByText('Rate Movies')).toBeInTheDocument();
        expect(screen.getByText('Get Recommendations')).toBeInTheDocument();
    });

    it('has proper semantic structure', () => {
        render(<AppHeader />);

        const header = screen.getByRole('banner');
        expect(header).toBeInTheDocument();

        const heading = screen.getByRole('heading', { level: 1 });
        expect(heading).toBeInTheDocument();
    });

    it('shows mobile menu button when navigation items exist', () => {
        render(<AppHeader currentPage="rating" onNavigate={mockOnNavigate} />);

        const menuButton = screen.getByLabelText('Open menu');
        expect(menuButton).toBeInTheDocument();
    });
});