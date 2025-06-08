from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from api.utils.transcript_utils import (
    get_transcript, chunk_transcript, embed_chunks, find_relevant_chunks,
    generate_sections_with_timestamps, generate_rag_response
)
from api.utils.visual_pipeline_utils import create_vid_embeddings, visual_query

CHUNK_SIZE = 500  # Can be adjusted based on your needs
app = FastAPI()

origins = [
    "http://localhost:3000",  # Your frontend URL
    "http://localhost:8000",  # If your backend also serves a frontend
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, PUT, DELETE, OPTIONS)
    allow_headers=["*"],  # Allows all headers
)

class VideoRequest(BaseModel):
    youtube_url: str

class VisualQueryRequest(BaseModel):
    query_text: str

class TextQueryRequest(BaseModel):
    embedded_chunks: list[dict]
    user_query: str
    similarity_threshold: Optional[float] = 0.3
    top_k: Optional[int] = 3

# -------------------------------
# Video Processing Routes
# -------------------------------

@app.post("/create_vid_embeddings")
def create_vid_embeddings_api(req: VideoRequest):
    try:
        collection_name_with_vid_embeddings = create_vid_embeddings(req.youtube_url)
        return {
            "status": "processed",
            "chroma_collection_name": collection_name_with_vid_embeddings
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/visual_query")
def visual_query_api(req: VisualQueryRequest):
    try:
        matches = visual_query(req.query_text)
        return {"matches": matches}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------------
# Transcript Processing Routes
# -------------------------------

@app.post("/create_text_embeddings")
def create_text_embeddings(req: VideoRequest):
    try:
        transcript = get_transcript(req.youtube_url)
        if not transcript:
            raise HTTPException(status_code=404, detail="Transcript not found.")

        chunks = chunk_transcript(transcript, chunk_size=CHUNK_SIZE)
        embedded_chunks = embed_chunks(chunks)

        return embedded_chunks
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/text_query")
def handle_user_query(req: TextQueryRequest):
    try:
        relevant_chunks = find_relevant_chunks(
            embedded_chunks = req.embedded_chunks,
            user_query=req.user_query,
            similarity_threshold=req.similarity_threshold,
            top_k=req.top_k
        )

        if not relevant_chunks:
            return {"message": "No relevant transcript chunks found."}

        response = generate_rag_response(relevant_chunks, req.user_query)
        return {
            "chunks": relevant_chunks,
            "response": response.choices[0].message["content"]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/sections")
def get_video_sections(req: VideoRequest):
    try:

        transcript = get_transcript(req.youtube_url)
        if not transcript:
            raise HTTPException(status_code=404, detail="Transcript not found.")

        result = generate_sections_with_timestamps(transcript)
        if result is None:
            raise HTTPException(status_code=500, detail="Failed to generate sections.")
        
        print("generated timestamps")

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))