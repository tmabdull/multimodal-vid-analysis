from transcript_utils import get_transcript, chunk_transcript, embed_chunks,find_relevant_chunks, generate_sections_with_timestamps, generate_rag_response

# Example usage
video_url = "https://www.youtube.com/watch?v=SaSZdCauekg"
transcript = get_transcript(video_url)
# print(transcript)

chunks = chunk_transcript(transcript, 1000)

embeddings = embed_chunks(chunks)

# print(generate_sections_with_timestamps(transcript))

user_query = "when does he talk to the karate instructor?"
relevant_chunks = find_relevant_chunks(embeddings, user_query)

response = generate_rag_response(relevant_chunks,user_query)
print(response)
