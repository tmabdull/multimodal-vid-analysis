'use client';

import { useState, useRef, useEffect, FormEvent } from 'react';
import { VideoPlayer } from '@/components/VideoPlayer';
import { VideoAnalysis } from '@/components/VideoAnalysis';
import { ChatInterface } from '@/components/ChatInterface';
import { VisualQuery } from '@/components/VisualQuery';

interface Message {
  role: 'user' | 'assistant';
  content: string;
}

interface SectionFromBackend {
  timestamp: string;
  title: string;
  description: string;
}

interface Analysis {
  summary: string;
  sections: SectionFromBackend[];
}

export default function Home() {
  const [url, setUrl] = useState('');
  const [videoId, setVideoId] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isChatLoading, setIsChatLoading] = useState(false);
  const [embeddedChunks, setEmbeddedChunks] = useState<any[]>([]);
  const [isCreatingEmbeddings, setIsCreatingEmbeddings] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const videoRef = useRef<HTMLIFrameElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const hhmmssToSeconds = (hhmmss: string): number => {
    const parts = hhmmss.split(':').map(Number);
    if (parts.length === 3) {
      return parts[0] * 3600 + parts[1] * 60 + parts[2];
    } else if (parts.length === 2) {
      return parts[0] * 60 + parts[1];
    }
    return 0;
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const extractVideoId = (url: string) => {
    const regExp = /^.*(youtu.be\/|v\/|u\/\w\/|embed\/|watch\?v=|&v=)([^#&?]*).*/;
    const match = url.match(regExp);
    return match && match[2].length === 11 ? match[2] : null;
  };

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    const id = extractVideoId(url);
    if (!id) {
      alert('Please enter a valid YouTube URL');
      return;
    }

    setIsLoading(true);
    setVideoId(id);
    setAnalysis(null);
    setMessages([]);
    setEmbeddedChunks([]);

    try {
      // Get text embeddings
      const textEmbeddingsResponse = await fetch('http://localhost:8000/create_text_embeddings', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ youtube_url: url }),
      });

      if (!textEmbeddingsResponse.ok) {
        throw new Error('Failed to get text embeddings');
      }

      const chunks = await textEmbeddingsResponse.json();
      setEmbeddedChunks(chunks);

      // Get video sections
      const sectionsResponse = await fetch('http://localhost:8000/sections', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ youtube_url: url }),
      });

      if (!sectionsResponse.ok) {
        throw new Error('Failed to get video sections');
      }

      const sectionsData = await sectionsResponse.json();

      console.log('Received sections data:', sectionsData);

      // Format the analysis data
      setAnalysis({
        summary: sectionsData.summary || 'No summary available',
        sections: sectionsData.sections || [],
      });

      console.log('Analysis timestamps after setting state:', sectionsData.sections);

    } catch (error) {
      console.error('Error analyzing video:', error);
      alert('Failed to analyze video. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleCreateVideoEmbeddings = async () => {
    if (!url) {
      alert('Please enter a YouTube URL first');
      return;
    }

    setIsCreatingEmbeddings(true);
    try {
      const response = await fetch('http://localhost:8000/create_vid_embeddings', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ youtube_url: url }),
      });

      if (!response.ok) {
        throw new Error('Failed to create video embeddings');
      }

      const data = await response.json();
      alert('Video embeddings created successfully!');
    } catch (error) {
      console.error('Error creating video embeddings:', error);
      alert('Failed to create video embeddings. Please try again.');
    } finally {
      setIsCreatingEmbeddings(false);
    }
  };

  const handleTimestampClick = (start: string) => {
    const iframe = document.querySelector('iframe');
    if (iframe) {
      const startSeconds = hhmmssToSeconds(start);
      iframe.src = `https://www.youtube.com/embed/${videoId}?start=${startSeconds}&autoplay=1`;
    }
  };

  const handleSendMessage = async (e: FormEvent) => {
    e.preventDefault();
    if (!inputMessage.trim() || !videoId || embeddedChunks.length === 0) return;

    const newMessage: Message = { role: 'user', content: inputMessage };
    setMessages((prev: Message[]) => [...prev, newMessage]);
    setInputMessage('');
    setIsChatLoading(true);

    try {
      const response = await fetch('http://localhost:8000/text_query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          embedded_chunks: embeddedChunks,
          user_query: inputMessage,
          similarity_threshold: 0.3,
          top_k: 3
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to get response');
      }

      const data = await response.json();
      setMessages((prev: Message[]) => [...prev, { role: 'assistant', content: data.response }]);
    } catch (error) {
      console.error('Error sending message:', error);
      setMessages((prev: Message[]) => [...prev, { role: 'assistant', content: 'Sorry, I encountered an error. Please try again.' }]);
    } finally {
      setIsChatLoading(false);
    }
  };

  return (
    <main className="min-h-screen p-8">
      <form onSubmit={handleSubmit} className="mb-8">
        <input
          type="text"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          placeholder="Enter YouTube URL"
          className="w-full p-2 mb-2 rounded"
        />
        <button type="submit" className="w-full p-2 bg-blue-500 text-white rounded" disabled={isLoading}>
          {isLoading ? 'Loading...' : 'Load Video'}
        </button>
      </form>

      {videoId && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          <div>
            <VideoPlayer videoId={videoId} ref={videoRef} />
            {analysis && analysis.sections.length > 0 && <VideoAnalysis sections={analysis.sections.map(section => ({
              timestamp: String(section.timestamp),
              title: section.title,
              description: section.description
            }))} onTimestampClick={handleTimestampClick} />}
          </div>

          <div className="card">
            <h2>Chat</h2>
            <div className="chat-container">
              <div className="messages">
                {messages.map((message, index) => (
                  <div key={index} className={`message ${message.role}`}>
                    {message.content}
                  </div>
                ))}
                {isChatLoading && (
                  <div className="message assistant">
                    Thinking...
                  </div>
                )}
                <div ref={messagesEndRef} />
              </div>

              <form onSubmit={handleSendMessage} className="chat-input">
                <input
                  type="text"
                  value={inputMessage}
                  onChange={(e) => setInputMessage(e.target.value)}
                  placeholder="Ask a question about the video..."
                  disabled={isChatLoading}
                />
                <button type="submit" className="button" disabled={isChatLoading}>
                  Send
                </button>
              </form>
              <button
                onClick={handleCreateVideoEmbeddings}
                disabled={isCreatingEmbeddings || !url}
                className="button mt-4 w-full"
              >
                {isCreatingEmbeddings ? 'Creating Video Embeddings...' : 'Create Video Embeddings'}
              </button>
            </div>
          </div>
        </div>
      )}
    </main>
  );
}
