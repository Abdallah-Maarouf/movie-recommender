import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import CallToAction from '../CallToAction';

// Mock framer-motion
vi.mock('framer-motion', () => ({
    motion: {
        div: ({ children, ...props }) => <div {...props}>{children}</div>,
        span: ({ children, ...props }) => <span {...props}>{children}</span>,
    },
}));

describe('CallToAction', () => {
    const mockOnClick = vi.fn();

    beforeEach(() => {
        mockOnClick.mockClear();
    });

    it('renders with default text', () => {
        render(<CallToAction onClick={mockOnClick} />);

        expect(screen.getByText('Get Started')).toBeInTheDocument();
    });

    it('renders with custom text', () => {
        render(<CallToAction onClick={mockOnClick} text="Start Rating Movies" />);

        expect(screen.getByText('Start Rating Movies')).toBeInTheDocument();
    });

    it('calls onClick when clicked', () => {
        render(<CallToAction onClick={mockOnClick} />);

        const button = screen.getByRole('button');
        fireEvent.click(button);

        expect(mockOnClick).toHaveBeenCalledTimes(1);
    });

    it('shows loading state when isLoading is true', () => {
        render(
            <CallToAction
                onClick={mockOnClick}
                isLoading={true}
                loadingText="Loading..."
            />
        );

        expect(screen.getByText('Loading...')).toBeInTheDocument();
    });

    it('is disabled when loading', () => {
        render(<CallToAction onClick={mockOnClick} isLoading={true} />);

        const button = screen.getByRole('button');
        expect(button).toBeDisabled();
    });

    it('has proper accessibility attributes', () => {
        render(
            <CallToAction
                onClick={mockOnClick}
                text="Start Rating"
                isLoading={false}
            />
        );

        const button = screen.getByRole('button');
        expect(button).toHaveAttribute('aria-label', 'Start Rating');
    });

    it('has loading aria-label when loading', () => {
        render(
            <CallToAction
                onClick={mockOnClick}
                isLoading={true}
                loadingText="Preparing..."
            />
        );

        const button = screen.getByRole('button');
        expect(button).toHaveAttribute('aria-label', 'Preparing...');
    });

    it('applies custom className', () => {
        render(
            <CallToAction
                onClick={mockOnClick}
                className="custom-class"
            />
        );

        const button = screen.getByRole('button');
        expect(button).toHaveClass('custom-class');
    });
});