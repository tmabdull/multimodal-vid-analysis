import { forwardRef } from 'react';

interface VideoPlayerProps {
  videoId: string;
}

export const VideoPlayer = forwardRef<HTMLIFrameElement, VideoPlayerProps>(
  ({ videoId }, ref) => {
    return (
      <div className="relative w-full pt-[56.25%] bg-black rounded-lg overflow-hidden">
        <iframe
          ref={ref}
          className="absolute top-0 left-0 w-full h-full"
          src={`https://www.youtube.com/embed/${videoId}`}
          title="YouTube video player"
          allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
          allowFullScreen
        />
      </div>
    );
  }
);

VideoPlayer.displayName = 'VideoPlayer'; 