interface Section {
  timestamp: string;
  title: string;
  description: string;
}

interface VideoAnalysisProps {
  sections: Section[];
  onTimestampClick: (timestamp: string) => void;
}

export const VideoAnalysis = ({ sections, onTimestampClick }: VideoAnalysisProps) => {
  return (
    <div className="bg-gray-800 rounded-lg p-6">
      <h2 className="text-2xl font-semibold mb-4">Video Sections</h2>
      <div className="space-y-4">
        {sections.map((section, index) => (
          <div key={index} className="bg-gray-700 rounded-lg p-4">
            <button
              onClick={() => onTimestampClick(section.timestamp)}
              className="text-blue-400 hover:text-blue-300 font-mono mb-2"
            >
              {section.timestamp}
            </button>
            <h3 className="text-lg font-medium mb-2">{section.title}</h3>
            <p className="text-gray-300">{section.description}</p>
          </div>
        ))}
      </div>
    </div>
  );
}; 