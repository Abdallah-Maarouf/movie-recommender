# Movie Recommender Frontend

A modern React frontend for the Movie Recommendation System with Netflix-style design.

## 🚀 Features

- **Modern React Setup**: Built with Vite for fast development and optimized builds
- **Netflix-Style Design**: Dark theme with red accents matching Netflix's visual identity
- **Responsive Design**: Mobile-first approach with Tailwind CSS
- **Smooth Animations**: Framer Motion for engaging user interactions
- **Component Library**: Reusable UI components with consistent styling
- **API Integration**: Axios-based service layer with error handling and retry logic
- **Performance Optimized**: Code splitting, lazy loading, and optimized builds

## 🛠️ Tech Stack

- **React 18** - Modern React with hooks and concurrent features
- **Vite** - Fast build tool and development server
- **Tailwind CSS** - Utility-first CSS framework
- **Framer Motion** - Animation library for React
- **Axios** - HTTP client for API communication
- **ESLint** - Code linting and formatting

## 📁 Project Structure

```
frontend/
├── src/
│   ├── components/          # Reusable UI components
│   │   ├── ui/             # Basic UI elements (Button, Card, etc.)
│   │   ├── movie/          # Movie-specific components
│   │   └── layout/         # Layout components
│   ├── pages/              # Page components
│   ├── hooks/              # Custom React hooks
│   ├── utils/              # Utility functions
│   ├── services/           # API services
│   ├── styles/             # Global styles and Tailwind config
│   └── assets/             # Images, icons, fonts
├── public/                 # Static assets
└── tests/                  # Test files
```

## 🎨 Design System

### Colors
- **Netflix Black**: `#141414` - Primary background
- **Netflix Dark Gray**: `#181818` - Card backgrounds
- **Netflix Gray**: `#333333` - Secondary elements
- **Netflix Light Gray**: `#B3B3B3` - Text and borders
- **Netflix Red**: `#E50914` - Primary accent color
- **Netflix White**: `#FFFFFF` - Primary text

### Typography
- **Font Family**: Helvetica Neue, Helvetica, Arial, sans-serif
- **Headings**: Bold weights with proper hierarchy
- **Body Text**: Regular weight with good readability

### Components
- **Buttons**: Primary (red), Secondary (gray), Outline, Ghost variants
- **Cards**: Dark background with hover animations
- **Loading States**: Spinners and skeleton loaders
- **Progress Bars**: Linear and circular variants

## 🚀 Getting Started

### Prerequisites
- Node.js 16+ 
- npm or yarn

### Installation

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm run dev
   ```

4. Open your browser and visit `http://localhost:3000`

### Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build
- `npm run lint` - Run ESLint

## 🔧 Configuration

### Environment Variables
Create a `.env` file in the frontend directory:

```env
VITE_API_URL=http://localhost:8000
VITE_NODE_ENV=development
VITE_DEBUG=true
```

### Tailwind Configuration
The Tailwind config includes custom Netflix colors, animations, and responsive breakpoints. See `tailwind.config.js` for details.

### API Configuration
API services are configured in `src/services/api.js` with:
- Base URL configuration
- Request/response interceptors
- Error handling and retry logic
- Timeout settings

## 📱 Responsive Design

The application is built with a mobile-first approach:

- **Mobile**: 1 column layout, touch-friendly controls
- **Tablet**: 2 column layout, optimized spacing
- **Desktop**: 3-4 column layout, hover effects
- **Large Desktop**: 4+ column layout, enhanced animations

## 🎭 Animations

Framer Motion is used for:
- Page transitions
- Component hover effects
- Loading animations
- Gesture handling on mobile
- Smooth state transitions

## 🧪 Testing

Testing setup includes:
- Component rendering tests
- Utility function tests
- API service tests with mocking
- Responsive design testing
- Accessibility testing

## 🚀 Deployment

### Build for Production
```bash
npm run build
```

The build artifacts will be stored in the `dist/` directory.

### Performance Optimizations
- Code splitting by route and vendor libraries
- Image optimization and lazy loading
- CSS purging and minification
- JavaScript minification and tree shaking
- Service worker for caching (future enhancement)

## 🤝 Contributing

1. Follow the existing code style and conventions
2. Use the established component patterns
3. Ensure responsive design across all screen sizes
4. Add appropriate animations and transitions
5. Test on multiple devices and browsers

## 📄 License

This project is part of the Movie Recommendation System.