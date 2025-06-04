import torch
import os
import re
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# Choose a model; "openai/clip-vit-base-patch16" is common and lightweight
model_name = "openai/clip-vit-base-patch16"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name, use_fast=True)
model.eval()  # Set to evaluation mode

def extract_timestamp_from_filename(filename):
    # Matches 'frame_0007s_processed.jpg' or 'frame_0007s.jpg'
    match = re.search(r'frame_(\d+)s', filename)
    if match:
        return int(match.group(1))
    return None

vid_id = "M_uPKpvf918"
frame_dir = f"video_data/{vid_id}/frames"
frame_files = [f for f in os.listdir(frame_dir) if f.endswith("_processed.jpg")]

frame_embeddings = []
for fname in frame_files:
    timestamp = extract_timestamp_from_filename(fname)
    frame_path = os.path.join(frame_dir, fname)
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

print("Frame Embeddings[0]:", frame_embeddings[0])

import chromadb

# Example: storing in Chroma
client = chromadb.Client()
collection = client.get_or_create_collection("video_frame_embeddings")

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

print("Stored in Chroma")