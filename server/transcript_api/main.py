from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional

from utils.transcript_utils import (
    get_transcript, chunk_transcript, embed_chunks, find_relevant_chunks,
    generate_sections_with_timestamps, generate_rag_response
)

app = FastAPI()

CHUNK_SIZE = 1000  # Can be adjusted based on your needs

# -------------------------------
# Request/Response Models
# -------------------------------

class QueryRequest(BaseModel):
    video_url: str
    user_query: str
    similarity_threshold: Optional[float] = 0.3
    top_k: Optional[int] = 3

class SectionRequest(BaseModel):
    video_url: str

# -------------------------------
# /query Route
# -------------------------------

@app.post("/query")
async def handle_user_query(payload: QueryRequest):
    try:
        transcript = get_transcript(payload.video_url)
        if not transcript:
            raise HTTPException(status_code=404, detail="Transcript not found.")

        chunks = chunk_transcript(transcript, chunk_size=CHUNK_SIZE)
        embedded_chunks = embed_chunks(chunks)
        relevant_chunks = find_relevant_chunks(
            embedded_chunks,
            user_query=payload.user_query,
            similarity_threshold=payload.similarity_threshold,
            top_k=payload.top_k
        )

        if not relevant_chunks:
            return {"message": "No relevant transcript chunks found."}

        response = generate_rag_response(relevant_chunks, payload.user_query)
        return {
            "chunks": relevant_chunks,
            "response": response.choices[0].message["content"]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------------
# /sections Route
# -------------------------------

@app.post("/sections")
async def get_video_sections(payload: SectionRequest):
    try:
        print("getting transcript")

        transcript = get_transcript(payload.video_url)
        if not transcript:
            raise HTTPException(status_code=404, detail="Transcript not found.")

        print("retireved transcript")

        print("generating timestamps")

        result = generate_sections_with_timestamps(transcript)
        if result is None:
            raise HTTPException(status_code=500, detail="Failed to generate sections.")
        
        print("generated timestamps")

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
