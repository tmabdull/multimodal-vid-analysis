import yt_dlp
import re
import torch
import chromadb
import cv2
import os
import numpy as np
from pathlib import Path
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# Video Processing
def extract_video_id(youtube_url):
    """
    Extracts the video ID from a YouTube URL.
    """
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
        # 'quiet': True,
        # 'no_warnings': True,
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

def extract_raw_frames(video_url, output_dir, interval_seconds=5):
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

def preprocess_frames(raw_frames, target_size=(224, 224)):
    """Normalize and resize frames for consistent processing"""
    processed_frames = []
    
    for frame_info in raw_frames:
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

def filter_significant_frames(processed_frames, threshold=0.3):
    """Apply motion detection to focus on relevant visual content"""
    scene_frames = []
    prev_frame = None

    for frame_info in processed_frames:
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
    
    # Filter to keep only significant frames (scene changes or regular intervals)
    significant_frames = [
        f for i, f in enumerate(scene_frames)
        if f['scene_change'] or i % 3 == 0
    ]  # Every 3rd frame + scene changes
    
    return significant_frames

# Embeddings
def create_vid_embeddings(vid_id, filtered_frames):
    # Choose a model; "openai/clip-vit-base-patch16" is common and lightweight
    model_name = "openai/clip-vit-base-patch16"
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name, use_fast=True) # use_fast requires PyTorch
    model.eval()

    frame_embeddings = []

    for frame_info in filtered_frames:
        timestamp = frame_info['timestamp']
        frame_path = frame_info['processed_path']

        image = Image.open(frame_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")

        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            embedding = image_features.cpu().numpy()[0]

        frame_embeddings.append({
            'embedding': embedding,
            'timestamp': timestamp,
            'frame_path': frame_path
        })

    # print("Frame Embeddings[0]:", frame_embeddings[0])

    return frame_embeddings

def store_vid_embeddings(vid_id, frame_embeddings, collection_name):
    client = chromadb.Client()
    collection = client.get_or_create_collection(collection_name)

    for frame in frame_embeddings:
        collection.add(
            embeddings=[frame['embedding']],
            metadatas=[{
                'timestamp': frame['timestamp'],
                'frame_path': frame['frame_path'],
                'video_id': vid_id
            }],
            ids=[f"{vid_id}_{int(frame['timestamp'])}"]
        )

    print("Embeddings stored in chroma collection:", collection_name)
    return collection_name

# Main visual processing function to be called by API route
def process_video_visuals(youtube_url, output_dir="./video_data", 
                        #   frame_interval_seconds=None, 
                        #   processed_img_size=None, motion_detect_threshold=None, 
                          chroma_collection_name="video_frame_embeddings"):
    '''
    Main logic for processing a YT vid, creating frame embeddings, 
    and storing embeddings into Chroma DB for future query comparisons
    '''
    metadata = get_video_metadata(youtube_url)
    vid_id = metadata['video_id']
    if not vid_id:
        print("WARNING: No Vid ID found")

    frame_output_dir = os.path.join(output_dir, vid_id)

    extracted_frames = extract_raw_frames(youtube_url, frame_output_dir)
    processed_frames = preprocess_frames(extracted_frames)
    significant_frames = filter_significant_frames(processed_frames)
    
    print(f"Processed video: {metadata}")
    print(f"Num Raw frames extracted: {len(extracted_frames)}")    
    print(f"Num Filtered/Processed frames: {len(significant_frames)}")    
    
    frame_embeddings = create_vid_embeddings(vid_id, significant_frames)
    store_vid_embeddings(vid_id, frame_embeddings, chroma_collection_name)

    return 0

# TODO: Natural Language Queries and Similarity Searches

if __name__ == "__main__":
    youtube_url = "https://www.youtube.com/watch?v=M_uPKpvf918"
    # youtube_url = "https://www.youtube.com/watch?v=SaSZdCauekg"

    process_video_visuals(youtube_url)
