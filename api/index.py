from .utils.visual_pipeline_utils import create_vid_embeddings, visual_query
from fastapi import FastAPI, Request
from pydantic import BaseModel

app = FastAPI()

class VideoRequest(BaseModel):
    youtube_url: str

class QueryRequest(BaseModel):
    query_text: str

@app.post("/create_vid_embeddings")
def create_vid_embeddings_api(req: VideoRequest):
    # Call your Python function
    collection_name_with_vid_embeddings = create_vid_embeddings(req.youtube_url, loaded=True)
    return {
        "status": "processed",
        "chroma_collection_name": collection_name_with_vid_embeddings
    }

@app.post("/visual_query")
def visual_query_api(req: QueryRequest):
    matches = visual_query(req.query_text)
    return {"matches": matches}
