# VibeVoice Frontend

This is the web frontend for VibeVoice, a speech generation model built with Next.js, React, and TypeScript.

## Tech Stack

- **Framework**: Next.js 15.5 (App Router)
- **Language**: TypeScript 5
- **Styling**: Tailwind CSS 4
- **UI Library**: React 19
- **Package Manager**: npm

## Project Structure

```
frontend/
├── app/              # Next.js app router pages and layouts
├── components/       # Reusable React components
├── hooks/           # Custom React hooks
├── lib/             # Utility functions and shared logic
├── types/           # TypeScript type definitions
├── api/             # API integration and client functions
├── public/          # Static assets
└── ...config files
```

## Getting Started

### Prerequisites

- Node.js 20+ (recommended)
- npm

### Installation

```bash
npm install
```

### Development

Run the development server with Turbopack:

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) to view the application.

### Build

Build the application for production:

```bash
npm run build
```

### Start Production Server

```bash
npm start
```

### Linting

```bash
npm run lint
```

## Directory Guidelines

- **`app/`**: App Router pages, layouts, and route handlers
- **`components/`**: Reusable UI components (consider organizing by feature or atomic design)
- **`hooks/`**: Custom React hooks for shared logic
- **`lib/`**: Utility functions, helpers, and shared business logic
- **`types/`**: TypeScript interfaces and type definitions
- **`api/`**: API client functions and integration with VibeVoice backend
- **`public/`**: Static files served directly (images, fonts, etc.)

## Development Notes

- Uses Turbopack for faster development builds
- Configured with TypeScript strict mode
- ESLint configured with Next.js recommended rules
- Tailwind CSS with PostCSS for styling
- Path aliases configured: `@/*` maps to project root

## Learn More

- [Next.js Documentation](https://nextjs.org/docs)
- [React Documentation](https://react.dev)
- [TypeScript Documentation](https://www.typescriptlang.org/docs)
- [Tailwind CSS Documentation](https://tailwindcss.com/docs)
