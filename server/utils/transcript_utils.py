import os
from dotenv import load_dotenv
import yt_dlp
import requests
import datetime
import openai
import numpy as np
import json

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

def seconds_to_hhmmss(seconds):
    return str(datetime.timedelta(seconds=int(seconds)))

def get_transcript(video_url):
    try:
        ydl_opts = {
            'skip_download': True,
            'writesubtitles': True,
            'writeautomaticsub': True,
            'subtitlesformat': 'json3',
            'quiet': True,
            'no_warnings': True
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=False)
            auto_captions = info.get('automatic_captions', {})

            if 'en' not in auto_captions:
                print("No English auto-captions found.")
                return []

            transcript_url = auto_captions['en'][0].get('url')
            if not transcript_url:
                print("Transcript URL not found.")
                return []

            response = requests.get(transcript_url)
            if response.status_code != 200:
                print(f"Failed to fetch transcript. HTTP {response.status_code}")
                return []

            transcript_data = response.json()
            events = transcript_data.get('events', [])
            transcript_list = []

            for event in events:
                if 'segs' in event and 'tStartMs' in event:
                    text = ''.join(seg['utf8'] for seg in event['segs']).strip()
                    if text:
                        start_sec = int(event['tStartMs']) / 1000
                        timestamp = seconds_to_hhmmss(start_sec)
                        transcript_list.append({
                            "text": text,
                            "start": timestamp,
                            "start_seconds": start_sec,
                            "video_url": video_url
                        })

            return transcript_list
    except Exception as e:
        print(f"[get_transcript] Error: {e}")
        return []

def chunk_transcript(transcript, chunk_size):
    chunks = []
    current_chunk = ""
    current_start = None

    for item in transcript:
        text = item["text"]
        # start = item["start"]
        start_seconds = item["start_seconds"]

        if current_start is None:
            # current_start = start
            current_start_seconds = start_seconds

        if len(current_chunk) + len(text) <= chunk_size:
            current_chunk += " " + text
        else:
            chunks.append({
                "text": current_chunk.strip(),
                # "start": current_start,
                "start_seconds": current_start_seconds
            })
            current_chunk = text
            # current_start = start
            current_start_seconds = start_seconds

    if current_chunk:
        chunks.append({
            "text": current_chunk.strip(),
            # "start": current_start,
            "start_seconds": current_start_seconds
        })

    return chunks

def embed_chunks(chunks):
    results = []

    for chunk in chunks:
        text = chunk["text"]
        start_seconds = chunk["start_seconds"]

        try:
            response = openai.Embedding.create(
                input=text,
                model="text-embedding-3-small"  # or "text-embedding-ada-002"
            )
            embedding = response['data'][0]['embedding']
            results.append({
                "embedding": embedding,
                "start_seconds": start_seconds,
                "text": text
            })

        except Exception as e:
            print(f"Error embedding chunk starting at {start_seconds}: {e}")

    return results


# computs cosine similarity between two vectors, the user query and transcirpt embedding
def cosine_similarity(a,b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a,b) / (np.linalg.norm(a) * np.linalg.norm(b))
    

def find_relevant_chunks(embeddings, user_query, top_k=3, similarity_threshold=0.30):
    # Return top_k most relevant chunks that exceed the similarity threshold
    # Step 1: Embed the user query
    
    query_response = openai.Embedding.create(
        input=user_query,
        model="text-embedding-3-small"
    )
    query_embedding = np.array(query_response["data"][0]["embedding"])

    # Step 2: Score all chunks by cosine similarity
    scored_embeddings = []
    for entry in embeddings:
        score = cosine_similarity(query_embedding, entry["embedding"])
        if score >= similarity_threshold:
            scored_embeddings.append((score, entry))

    # Step 3: Sort by descending similarity
    scored_embeddings.sort(key=lambda x: x[0], reverse=True)

    # Step 4: Return top_k matches
    top_chunks = scored_embeddings[:top_k]

    return [
        {
            "text": chunk["text"],
            "start_seconds": chunk["start_seconds"],
            "score": score
        }
        for score, chunk in top_chunks
    ]

def generate_sections_with_timestamps(transcript_list):
    """
    Generates a summary and key sections with timestamps and descriptions
    from a transcript list using the OpenAI API.

    Args:
        transcript_list: A list of transcript segments, each a dictionary
                         with 'text' and 'start_seconds'.

    Returns:
        A dictionary containing 'summary' and a list of 'sections',
        or None if an error occurs.
    """

    if not transcript_list:
        print("Transcript list is empty")
        return None
    
    # Combine transcript segments into a single text
    full_transcript_text = " ".join([item['text'] for item in transcript_list])

    # Find the start time for each segment to reference in the prompt if needed
    # Create a mapping from text snippet to its start time
    text_to_start_time = {item['text']: item['start_seconds'] for item in transcript_list}

    prompt = f"""
    Analyze the following video transcript and provide a JSON response with:
    1. A comprehensive summary of the entire transcript.
    2. Key sections or topics covered, with their estimated start timestamps (in seconds) and a brief description.
    3. The timestamps should correspond to the start of the relevant discussion in the transcript.

    Format the response strictly as a JSON object with two keys: "summary" (string) and "sections" (an array of objects).
    Each object in the "sections" array should have the keys "timestamp" (number, in seconds), "title" (string, a brief title for the section), and "description" (string, a brief description of the section's content).

    Transcript:
    {full_transcript_text}

    JSON Response:
    """

    try:
        response = openai.ChatCompletion.create(
            model = "gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that analyzes video transcripts."},
                {"role": "user", "content": prompt}
            ],
            response_format = {"type": "json_object"} # Request JSON object response
        )

        # Extract and parse the JSON content from the response
        response_content = response.choices[0].message['content']
        sections_data = json.loads(response_content)

        processed_sections = []
        for section in sections_data.get('sections', []):
             # Find the transcript segment with the start_seconds closest to the LLM's estimated timestamp
             closest_segment = min(transcript_list,
                                   key=lambda item: abs(item['start_seconds'] - section.get('timestamp', 0)))
             processed_sections.append({
                 "timestamp": closest_segment['start_seconds'],
                 "title": section.get('title', 'Section'),
                 "description": section.get('description', 'No description provided.')
             })

        return {
            "summary": sections_data.get('summary', 'No summary provided.'),
            "sections": processed_sections
        }

    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON response from API: {e}")
        print(f"API response content: {response_content}")
        return None
    except Exception as e:
        print(f"Error generating sections: {e}")
        return None


def generate_rag_response(relevant_chunks,user_prompt):

    context = "\n\n".join(
        f"[{seconds_to_hhmmss(c['start_seconds'])}] {c['text']}" for c in relevant_chunks
    )

    system_prompt = f"""
    
    you are a multimodal video analysis agent. You are expect to answer user questions about what is said in the video.

    You will be provide text chunks of the transcript and their start times in the video as context. These chunks are the most relavent to the user query.

    Provide a concise yet helpful answer. If your answer references a specific part in the video, include a timestamp if available in the following format:
    
    [HH:MM:SS]

    Context: {relevant_chunks}
    Answer:
    """

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ], 
        temperature=0.7
    )

    return response