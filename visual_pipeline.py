import yt_dlp
import re
import cv2
import os
import numpy as np
from pathlib import Path

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

def extract_raw_frames(youtube_url, duration, output_dir, interval_seconds=None):
    """Extract frames at regular intervals using OpenCV"""
    
    # Create output directory for downloaded frames
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # yt-dlp download options: limit to 720p, video only (no audio)
    ydl_opts = {
        # 'download_sections': [{'section': {'start_time': 45, 'end_time': 115}}],
        'format': 'bestvideo[height<=720][ext=mp4]/best[height<=720][ext=mp4]/bestvideo+bestaudio/best',
        'outtmpl': 'temp_video.%(ext)s',
        'retries': 3,
        'fragment_retries': 3
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])

    # Find downloaded file
    temp_files = [f for f in os.listdir('.') if f.startswith('temp_video') and f.endswith('.mp4')]
    if not temp_files:
        raise FileNotFoundError("Video download failed or file not found.")
    temp_video_path = temp_files[0]

    # Get vid properties with OpenCV
    cap = cv2.VideoCapture(temp_video_path)
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video file {temp_video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if not interval_seconds:
        if duration < (6 * 60):
            interval_seconds = 5
        elif duration < (20 * 60):
            interval_seconds = 10
        elif duration < (60 * 60):
            interval_seconds = 20
        else:
            interval_seconds = 60

    print(f"Duration (secs): {duration}, Frame Intervals (secs): {interval_seconds}")
    frame_interval = int(fps * interval_seconds) if fps else 1
    
    extracted_frames = []
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Extract frame at specified intervals
        if frame_count % frame_interval == 0:
            timestamp = frame_count / fps if fps else 0
            raw_frame_filename = f"frame_{int(timestamp):04d}s.jpg"
            raw_frame_path = os.path.join(output_dir, raw_frame_filename)
            
            cv2.imwrite(raw_frame_path, frame)
            
            extracted_frames.append({
                'timestamp': timestamp,
                'raw_frame_path': raw_frame_path,
                'frame_number': frame_count
            })
            
        frame_count += 1
    
    cap.release()
    
    # Clean up temporary video file
    os.remove(temp_video_path)
    
    return extracted_frames

def preprocess_frames(raw_frames, target_size=(224, 224)):
    """Normalize and resize frames for consistent processing"""
    processed_frames = []
    
    for frame_info in raw_frames:
        raw_frame_path = frame_info['raw_frame_path']
        
        # Read and preprocess frame
        frame = cv2.imread(raw_frame_path)
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
        processed_path = raw_frame_path.replace('.jpg', '_processed.jpg')
        cv2.imwrite(processed_path, denoised_frame)
        
        processed_frames.append({
            'timestamp': frame_info['timestamp'],
            'original_frame_path': raw_frame_path,
            'frame_path': processed_path,
            'frame_array': denoised_frame
        })
    
    return processed_frames

def filter_significant_frames(processed_frames, threshold=0.3):
    """Apply motion detection to focus on relevant visual content"""
    scene_frames = []
    prev_frame = None

    for frame_info in processed_frames:
        frame = cv2.imread(frame_info['frame_path'], cv2.IMREAD_GRAYSCALE)
        
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
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from typing import List
import numpy as np

class CLIPImageEmbeddings(Embeddings):
    """Custom CLIP image embedding class for LangChain integration"""
    def __init__(self, model_name="openai/clip-vit-base-patch16"):
        from transformers import CLIPModel, CLIPProcessor
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

    def embed_documents(self, image_paths: List[str]) -> List[List[float]]:
        """Embed a list of image paths using CLIP"""
        from PIL import Image
        import torch
        
        embeddings = []
        for path in image_paths:
            image = Image.open(path).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt")
            with torch.no_grad():
                features = self.model.get_image_features(**inputs)
                features = features / features.norm(p=2, dim=-1, keepdim=True)
                embeddings.append(features.cpu().numpy()[0].tolist())
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed text query using CLIP"""
        import torch
        inputs = self.processor(text=text, return_tensors="pt", padding=True)
        with torch.no_grad():
            features = self.model.get_text_features(**inputs)
            features = features / features.norm(p=2, dim=-1, keepdim=True)
            return features.cpu().numpy()[0].tolist()

def store_vid_embeddings(vid_id, frame_embeddings, collection_name):
    """Store embeddings using LangChain's Chroma integration"""
    # Create LangChain documents with metadata
    documents = [
        Document(
            page_content=frame['frame_path'],  # Use path as content
            metadata={
                'timestamp': frame['timestamp'],
                'video_id': vid_id,
                'frame_path': frame['frame_path']
            }
        ) for frame in frame_embeddings
    ]
    
    # Extract image paths for embedding
    image_paths = [frame['frame_path'] for frame in frame_embeddings]
    
    # Initialize Chroma with CLIP embeddings
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=CLIPImageEmbeddings(),
        collection_name=collection_name,
        persist_directory=None,  # In-memory for MVP
        collection_metadata={"hnsw:space": "cosine"}
    )
    
    print(f"Stored {len(documents)} embeddings in Chroma collection: {collection_name}")
    return vector_store

# Main querying function
def visual_query(query_text, max_k=20, similarity_threshold=0.3, 
                collection_name="video_frame_embeddings"):
    """Perform visual similarity search using LangChain"""
    # Initialize Chroma with CLIP embeddings
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=CLIPImageEmbeddings(),
        persist_directory=None,
        collection_metadata={"hnsw:space": "cosine"}
    )
    
    # Perform similarity search with threshold
    results = vector_store.similarity_search_with_relevance_scores(
        query=query_text,
        k=max_k,
        score_threshold=similarity_threshold
    )
    
    # Format results
    matches = [{
        'timestamp': doc.metadata['timestamp'],
        'frame_path': doc.metadata['frame_path'],
        'score': score
    } for doc, score in results]

    return matches

# Main visual processing + embedding function
def process_video_visuals(youtube_url, loaded=False, output_dir="./video_data",
                         chroma_collection_name="video_frame_embeddings"):
    '''Main video processing pipeline'''
    metadata = get_video_metadata(youtube_url)
    vid_id = metadata['video_id']
    
    frame_output_dir = os.path.join(output_dir, vid_id)

    if loaded:
        print("Already loaded - getting processed frames from disk...")
        # Load processed frames from disk
        processed_frames = []
        
        pattern = re.compile(r"frame_(\d+)s_processed\.jpg")
        for filename in os.listdir(frame_output_dir):
            match = pattern.match(filename)
            if match:
                timestamp = float(match.group(1))
                frame_path = os.path.join(frame_output_dir, filename)
                processed_frames.append({'frame_path': frame_path, 'timestamp': timestamp})
    else:
        print("No frames found in local. Downloading and processing instead...")
        raw_frames = extract_raw_frames(youtube_url, metadata['duration'], frame_output_dir)
        print(f"Num Raw frames extracted: {len(raw_frames)}")   
        processed_frames = preprocess_frames(raw_frames)
    
    significant_frames = filter_significant_frames(processed_frames)
    frames_to_embed = significant_frames
    frames_to_embed = processed_frames
    
    print(f"Processed video: {metadata}")
    print(f"Num Filtered/Processed frames: {len(frames_to_embed)}")   
    # print(f"Frame_to_embed 0: {frames_to_embed[0].keys()}") 
    
    # Create frame embeddings using CLIP
    print("Creating embeddings...")
    embedding_model = CLIPImageEmbeddings()
    image_paths = [frame['frame_path'] for frame in frames_to_embed]
    embeddings = embedding_model.embed_documents(image_paths)
    
    # Add embeddings to frame info
    frame_embeddings = [{
        **frame,
        'embedding': emb
    } for frame, emb in zip(frames_to_embed, embeddings)]
    
    # Store using LangChain Chroma
    print("Storing embeddings...")
    store_vid_embeddings(vid_id, frame_embeddings, chroma_collection_name)
    
    return 0

if __name__ == "__main__":
    youtube_url, query = "https://www.youtube.com/watch?v=M_uPKpvf918", "diagram"
    # youtube_url, query = "https://www.youtube.com/watch?v=yYFmYWpMPlE", "tech stack diagram"
    youtube_url, query = "https://www.youtube.com/watch?v=SaSZdCauekg", "teddy bear"

    process_video_visuals(youtube_url, loaded=True)
    print("Embeddings stored! Starting visual query...")
    matches = visual_query(query, max_k=1000, similarity_threshold=0.01)
    matches_sorted_by_score = sorted(matches, key=lambda x: x["score"])
    for m in matches_sorted_by_score:
        print(f"Score: {m['score']:.3f}, TS: {m['timestamp']:.0f}")
