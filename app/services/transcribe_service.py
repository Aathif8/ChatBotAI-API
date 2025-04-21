import openai
import io
import requests
import os
from dotenv import load_dotenv

load_dotenv()


HF_API_KEY = os.getenv("HF_API_KEY") 
HF_API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-base"
HF_TRANSCRIBE_URL = "https://api-inference.huggingface.co/models/openai/whisper-small"
HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"}

# Function to transcribe audio
def transcribe_audio(audio_bytes):
    audio_file = io.BytesIO(audio_bytes)
    audio_file.name = "audio.mp3"

    response = openai.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file
    )

    return response.text


# Speech-to-Text Function
def TranscribeHF(audio_bytes):
    response = requests.post(
        HF_TRANSCRIBE_URL,
        headers={
            "Authorization": f"Bearer {HF_API_KEY}",
            "Content-Type": "application/octet-stream"
        },
        data=audio_bytes
    )

    if response.status_code == 200:
        result = response.json()
        return result.get("text", "No text found.")
    else:
        return f"Error: {response.status_code} - {response.text}"