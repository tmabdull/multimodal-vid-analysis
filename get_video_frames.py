from youtube_transcript_api import YouTubeTranscriptApi
import yt_dlp

def extract_video_id(youtube_url):
    """
    Extracts the video ID from a YouTube URL.
    """
    import re
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
        r'(?:embed\/)([0-9A-Za-z_-]{11})',
        r'(?:watch\?v=)([0-9A-Za-z_-]{11})'
    ]
    for pattern in patterns:
        match = re.search(pattern, youtube_url)
        if match:
            return match.group(1)
    raise ValueError("Could not extract video ID from URL.")

def get_video_metadata(youtube_url):
    """Extract comprehensive video metadata"""
    video_id = extract_video_id(youtube_url)
    
    # Use yt-dlp for metadata extraction
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
        
    metadata = {
        'video_id': video_id,
        'title': info.get('title', ''),
        'duration': info.get('duration', 0),  # in seconds
        'upload_date': info.get('upload_date', ''),
        'view_count': info.get('view_count', 0),
        'channel': info.get('uploader', ''),
        'thumbnail': info.get('thumbnail', '')
    }
    
    return metadata

from youtube_transcript_api import YouTubeTranscriptApi

def get_transcript_with_timestamps(video_id):
    """
    Fetches the transcript and returns a list of dicts with text, start, and duration.
    """
    try:
        # This returns a list of dicts, each with 'text', 'start', 'duration'
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        for entry in transcript:
            # Each entry is a dict: {'text': ..., 'start': ..., 'duration': ...}
            entry['end'] = entry['start'] + entry['duration']
        return transcript
    except Exception as e:
        print(f"Error extracting transcript: {e}")
        return None

def format_timestamp(seconds):
    """Convert seconds to HH:MM:SS format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

import cv2
import os
import numpy as np
from pathlib import Path

def extract_frames_opencv(video_url, output_dir, interval_seconds=5):
    """Extract frames at regular intervals using OpenCV"""
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Download video temporarily or use direct URL if supported
    cap = cv2.VideoCapture(video_url)
    
    if not cap.isOpened():
        # If direct URL doesn't work, download video first
        import yt_dlp
        ydl_opts = {
            'format': 'best[height<=720]',  # Limit quality for processing
            'outtmpl': 'temp_video.%(ext)s'
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
            
        # Find downloaded file
        temp_files = [f for f in os.listdir('.') if f.startswith('temp_video')]
        if temp_files:
            cap = cv2.VideoCapture(temp_files[0])
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    frame_interval = int(fps * interval_seconds)
    extracted_frames = []
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Extract frame at specified intervals
        if frame_count % frame_interval == 0:
            timestamp = frame_count / fps
            frame_filename = f"frame_{int(timestamp):04d}s.jpg"
            frame_path = os.path.join(output_dir, frame_filename)
            
            cv2.imwrite(frame_path, frame)
            
            extracted_frames.append({
                'timestamp': timestamp,
                'frame_path': frame_path,
                'frame_number': frame_count
            })
            
        frame_count += 1
    
    cap.release()
    
    # Clean up temporary video file
    for temp_file in [f for f in os.listdir('.') if f.startswith('temp_video')]:
        os.remove(temp_file)
    
    return extracted_frames

def chunk_transcript_with_timestamps(transcript_data, chunk_size=1000, overlap=200):
    """Split transcript into chunks while preserving timestamp ranges"""
    chunks = []
    current_chunk = ""
    current_start_time = None
    current_end_time = None
    current_length = 0
    
    for entry in transcript_data:
        text = entry['text']
        start_time = entry['start_time']
        end_time = entry['end_time']
        
        # Initialize first chunk
        if current_start_time is None:
            current_start_time = start_time
        
        # Check if adding this text exceeds chunk size
        if current_length + len(text) > chunk_size and current_chunk:
            # Save current chunk
            chunks.append({
                'text': current_chunk.strip(),
                'start_time': current_start_time,
                'end_time': current_end_time,
                'chunk_index': len(chunks)
            })
            
            # Start new chunk with overlap
            overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
            current_chunk = overlap_text + " " + text
            current_start_time = start_time
            current_length = len(current_chunk)
        else:
            # Add to current chunk
            current_chunk += " " + text
            current_length += len(text)
        
        current_end_time = end_time
    
    # Add final chunk
    if current_chunk:
        chunks.append({
            'text': current_chunk.strip(),
            'start_time': current_start_time,
            'end_time': current_end_time,
            'chunk_index': len(chunks)
        })
    
    return chunks

def preprocess_frames(frame_paths, target_size=(224, 224)):
    """Normalize and resize frames for consistent processing"""
    processed_frames = []
    
    for frame_info in frame_paths:
        frame_path = frame_info['frame_path']
        
        # Read and preprocess frame
        frame = cv2.imread(frame_path)
        if frame is None:
            continue
            
        # Resize frame
        resized_frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
        
        # Normalize pixel values to [0, 1]
        normalized_frame = resized_frame.astype(np.float32) / 255.0
        
        # Apply noise reduction (optional)
        denoised_frame = cv2.bilateralFilter(
            (normalized_frame * 255).astype(np.uint8), 9, 75, 75
        )
        
        # Save processed frame
        processed_path = frame_path.replace('.jpg', '_processed.jpg')
        cv2.imwrite(processed_path, denoised_frame)
        
        processed_frames.append({
            'timestamp': frame_info['timestamp'],
            'original_path': frame_path,
            'processed_path': processed_path,
            'frame_array': denoised_frame
        })
    
    return processed_frames

def detect_scene_changes(frame_paths, threshold=0.3):
    """Apply motion detection to focus on relevant visual content"""
    scene_frames = []
    prev_frame = None

    for frame_info in frame_paths:
        frame = cv2.imread(frame_info['processed_path'], cv2.IMREAD_GRAYSCALE)
        
        if prev_frame is not None:
            # Calculate frame difference
            diff = cv2.absdiff(prev_frame, frame)
            mean_diff = np.mean(diff) / 255.0
            
            # If significant change detected, mark as scene change
            if mean_diff > threshold:
                scene_frames.append({
                    **frame_info,
                    'scene_change': True,
                    'change_magnitude': mean_diff
                })
            else:
                scene_frames.append({
                    **frame_info,
                    'scene_change': False,
                    'change_magnitude': mean_diff
                })
        else:
            # First frame is always a scene change
            scene_frames.append({
                **frame_info,
                'scene_change': True,
                'change_magnitude': 1.0
            })
        
        prev_frame = frame
    
    return scene_frames

def process_youtube_video_step1(youtube_url, output_dir="./video_data"):
    """Complete Step 1 processing pipeline"""
    
    print("Extracting video metadata...")
    metadata = get_video_metadata(youtube_url)
    video_id = metadata['video_id']

    print("Metadata:", metadata)
    
    # print("Downloading transcript...")
    # transcript = get_transcript_with_timestamps(video_id)
    # if transcript is None:
    #     raise Exception("Could not extract transcript from video")
    
    print("Extracting video frames...")
    frame_output_dir = os.path.join(output_dir, video_id, "frames")
    extracted_frames = extract_frames_opencv(youtube_url, frame_output_dir, interval_seconds=7)
    
    print("Preprocessing data...")
    # # Chunk transcript
    # transcript_chunks = chunk_transcript_with_timestamps(transcript)
    
    # Preprocess frames
    processed_frames = preprocess_frames(extracted_frames)
    
    # Detect scene changes for better frame selection
    scene_frames = detect_scene_changes(processed_frames)
    
    # Filter to keep only significant frames (scene changes or regular intervals)
    significant_frames = [
        f for i, f in enumerate(scene_frames)
        if f['scene_change'] or i % 3 == 0
    ]  # Every 3rd frame + scene changes
    
    return {
        'metadata': metadata,
        # 'transcript_chunks': transcript_chunks,
        'frames': significant_frames,
        # 'raw_transcript': transcript
    }

# Usage example
if __name__ == "__main__":
    youtube_url = "https://www.youtube.com/watch?v=M_uPKpvf918"
    result = process_youtube_video_step1(youtube_url)
    
    # print(f"Processed video: {result['metadata']['title']}")
    # print(f"Duration: {result['metadata']['duration']} seconds")
    # print(f"Transcript chunks: {len(result['transcript_chunks'])}")
    print(f"Extracted frames: {len(result['frames'])}")
