'use client';

import { useState, useRef, useEffect } from 'react';
import { VideoPlayer } from '@/components/VideoPlayer';
import { VideoAnalysis } from '@/components/VideoAnalysis';
import { ChatInterface } from '@/components/ChatInterface';
import { VisualQuery } from '@/components/VisualQuery';

interface Message {
  role: 'user' | 'assistant';
  content: string;
}

interface Timestamp {
  text: string;
  start: number;
  end: number;
}

interface Analysis {
  summary: string;
  key_points: string[];
  timestamps: Timestamp[];
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
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const videoRef = useRef<HTMLIFrameElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const extractVideoId = (url: string) => {
    const regExp = /^.*(youtu.be\/|v\/|u\/\w\/|embed\/|watch\?v=|&v=)([^#&?]*).*/;
    const match = url.match(regExp);
    return match && match[2].length === 11 ? match[2] : null;
  };

  const handleSubmit = async (e: React.FormEvent) => {
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
      // Create video embeddings
      await fetch('http://localhost:8000/create_vid_embeddings', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ youtube_url: url }),
      });

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
        key_points: sectionsData.key_points || [],
        timestamps: sectionsData.timestamps || [],
      });

      console.log('Analysis timestamps after setting state:', sectionsData.timestamps);

    } catch (error) {
      console.error('Error analyzing video:', error);
      alert('Failed to analyze video. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleTimestampClick = (start: number) => {
    const iframe = document.querySelector('iframe');
    if (iframe) {
      iframe.src = `https://www.youtube.com/embed/${videoId}?start=${start}&autoplay=1`;
    }
  };

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputMessage.trim() || !videoId || embeddedChunks.length === 0) return;

    const newMessage: Message = { role: 'user', content: inputMessage };
    setMessages(prev => [...prev, newMessage]);
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
      setMessages(prev => [...prev, { role: 'assistant', content: data.response }]);
    } catch (error) {
      console.error('Error sending message:', error);
      setMessages(prev => [...prev, { role: 'assistant', content: 'Sorry, I encountered an error. Please try again.' }]);
    } finally {
      setIsChatLoading(false);
    }
  };

  return (
    <main className="container">
      <div className="header">
        <h1>YouTube Video Analysis</h1>
        <p>Enter a YouTube URL to analyze the video content</p>
      </div>

      <form onSubmit={handleSubmit} className="input-group">
        <input
          type="text"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          placeholder="Enter YouTube URL"
          required
        />
        <button type="submit" className="button" disabled={isLoading}>
          {isLoading ? 'Analyzing...' : 'Analyze Video'}
        </button>
      </form>

      {videoId && (
        <div className="grid">
          <div className="card">
            <h2>Video</h2>
            <div className="video-container">
              <iframe
                src={`https://www.youtube.com/embed/${videoId}`}
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                allowFullScreen
              />
            </div>

            {analysis && (
              <>
                <div className="section">
                  <h3>Summary</h3>
                  <p>{analysis.summary}</p>
                </div>

                <div className="section">
                  <h3>Key Points</h3>
                  <ul>
                    {analysis.key_points.map((point, index) => (
                      <li key={index}>{point}</li>
                    ))}
                  </ul>
                </div>

                <div className="section">
                  <h3>Timestamps</h3>
                  <div>
                    {analysis.timestamps.map((timestamp, index) => (
                      <span
                        key={index}
                        className="timestamp"
                        onClick={() => handleTimestampClick(timestamp.start)}
                      >
                        {timestamp.text}
                      </span>
                    ))}
                  </div>
                </div>
              </>
            )}
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
            </div>
          </div>
        </div>
      )}
    </main>
  );
}
