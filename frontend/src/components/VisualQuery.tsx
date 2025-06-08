import { useState } from 'react';

interface VisualQueryProps {
  videoId: string;
}

export const VisualQuery = ({ videoId }: VisualQueryProps) => {
  const [query, setQuery] = useState('');
  const [matches, setMatches] = useState<any[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim() || isLoading) return;

    setIsLoading(true);
    try {
      const response = await fetch('http://localhost:8000/visual_query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query_text: query }),
      });

      const data = await response.json();
      setMatches(data.matches || []);
    } catch (error) {
      console.error('Error querying video:', error);
      setMatches([]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="bg-gray-800 rounded-lg p-6">
      <h2 className="text-2xl font-semibold mb-4">Visual Search</h2>
      
      <form onSubmit={handleSubmit} className="mb-4">
        <div className="flex gap-2">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Describe what you're looking for..."
            className="flex-1 px-4 py-2 rounded-lg bg-gray-700 border border-gray-600 focus:border-blue-500 focus:ring-2 focus:ring-blue-500 outline-none"
            disabled={isLoading}
          />
          <button
            type="submit"
            disabled={isLoading}
            className="px-6 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg font-semibold transition-colors disabled:opacity-50"
          >
            {isLoading ? 'Searching...' : 'Search'}
          </button>
        </div>
      </form>

      {matches.length > 0 && (
        <div className="space-y-4">
          <h3 className="text-lg font-medium">Matching Clips</h3>
          <div className="grid grid-cols-2 gap-4">
            {matches.map((match, index) => (
              <div key={index} className="bg-gray-700 rounded-lg p-4">
                <p className="text-sm text-gray-300 mb-2">Timestamp: {match.timestamp}</p>
                <p className="text-white">{match.description}</p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}; 