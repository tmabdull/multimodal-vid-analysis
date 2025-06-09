import { useState } from 'react';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  timestamps?: string[];
}

interface ChatInterfaceProps {
  embeddedChunks: any[];
  onTimestampClick: (timestamp: string) => void;
}

export const ChatInterface = ({ embeddedChunks, onTimestampClick }: ChatInterfaceProps) => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage: Message = { role: 'user', content: input };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await fetch('http://localhost:8000/text_query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          embeddings: embeddedChunks,
          user_query: input,
        }),
      });

      const data = await response.json();
      
      const assistantMessage: Message = {
        role: 'assistant',
        content: data.response,
        timestamps: data.chunks?.map((chunk: any) => chunk.timestamp) || [],
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Error sending message:', error);
      setMessages(prev => [
        ...prev,
        { role: 'assistant', content: 'Sorry, there was an error processing your request.' },
      ]);
    } finally {
      setIsLoading(false);
    }
  };
  return (
    <div className="bg-gray-800 rounded-lg h-[600px] flex flex-col p-6">
      <h2 className="text-2xl font-semibold mb-4">Chat with Video</h2>
  
      {/* Scrollable message list */}
      <div className="flex-1 overflow-y-auto space-y-4">
        {messages.map((message, index) => (
          <div
            key={index}
            className={`p-4 rounded-lg ${
              message.role === 'user' ? 'bg-blue-600 ml-8' : 'bg-gray-700 mr-8'
            }`}
          >
            <p className="text-white">{message.content}</p>
            {message.timestamps && message.timestamps.length > 0 && (
              <div className="mt-2 flex flex-wrap gap-2">
                {message.timestamps.map((timestamp, i) => (
                  <button
                    key={i}
                    onClick={() => onTimestampClick(timestamp)}
                    className="text-sm text-blue-300 hover:text-blue-200"
                  >
                    {timestamp}
                  </button>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>
  
      {/* Input bar pinned to the bottom */}
      <form onSubmit={handleSubmit} className="mt-4 flex gap-2">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask about the video..."
          className="flex-1 px-4 py-2 rounded-lg bg-gray-700 border border-gray-600 focus:border-blue-500 focus:ring-2 focus:ring-blue-500 outline-none"
          disabled={isLoading}
        />
        <button
          type="submit"
          disabled={isLoading}
          className="px-6 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg font-semibold transition-colors disabled:opacity-50"
        >
          {isLoading ? 'Sending...' : 'Send'}
        </button>
      </form>
    </div>
  );  
}; 