# Video Analysis Frontend

This is the frontend application for the Video Analysis Platform. It provides a modern, user-friendly interface for analyzing YouTube videos, including features for:

- Video playback with clickable timestamps
- Section breakdown with timestamps
- Chat interface for asking questions about the video content
- Visual search capabilities

## Setup

1. Install dependencies:
```bash
npm install
```

2. Start the development server:
```bash
npm run dev
```

3. Make sure the backend server is running on `http://localhost:8000`

## Features

- **Video Analysis**: Upload a YouTube video URL to analyze its content
- **Section Breakdown**: View a detailed breakdown of the video with clickable timestamps
- **Chat Interface**: Ask questions about the video content and get AI-powered responses
- **Visual Search**: Search for specific visual elements in the video using natural language

## Technologies Used

- Next.js 14
- React 18
- TypeScript
- Tailwind CSS
- OpenAI API (v0.28.1)

## Getting Started

First, run the development server:

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
# or
bun dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

You can start editing the page by modifying `app/page.tsx`. The page auto-updates as you edit the file.

This project uses [`next/font`](https://nextjs.org/docs/app/building-your-application/optimizing/fonts) to automatically optimize and load [Geist](https://vercel.com/font), a new font family for Vercel.

## Learn More

To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.

You can check out [the Next.js GitHub repository](https://github.com/vercel/next.js) - your feedback and contributions are welcome!

## Deploy on Vercel

The easiest way to deploy your Next.js app is to use the [Vercel Platform](https://vercel.com/new?utm_medium=default-template&filter=next.js&utm_source=create-next-app&utm_campaign=create-next-app-readme) from the creators of Next.js.

Check out our [Next.js deployment documentation](https://nextjs.org/docs/app/building-your-application/deploying) for more details.
