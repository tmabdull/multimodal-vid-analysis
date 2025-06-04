from transcript_utils import get_transcript_dict, chunk_transcript, embed_chunks

# Example usage
video_url = "https://www.youtube.com/watch?v=SaSZdCauekg"
transcript = get_transcript_dict(video_url)
# print(transcript)

chunks = chunk_transcript(transcript, 1000)

embeddings = embed_chunks(chunks)

print(embeddings[0])

