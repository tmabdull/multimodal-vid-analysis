import os
from dotenv import load_dotenv
import yt_dlp
import requests
import datetime
import openai

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

def seconds_to_hhmmss(seconds):
    return str(datetime.timedelta(seconds=int(seconds)))

def get_transcript_dict(video_url):
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
            return {}

        transcript_url = auto_captions['en'][0].get('url')
        if not transcript_url:
            print("No transcript URL found.")
            return {}

        response = requests.get(transcript_url)
        if response.status_code != 200:
            print(f"Failed to download transcript: {response.status_code}")
            return {}

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
            })

        except Exception as e:
            print(f"Error embedding chunk starting at {start_seconds}: {e}")

    return results


