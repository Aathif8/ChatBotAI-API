import requests
import os
from dotenv import load_dotenv

load_dotenv()

HF_API_KEY = os.getenv("HF_API_KEY")
HF_API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-base"
HF_TRANSCRIBE_URL = "https://api-inference.huggingface.co/models/openai/whisper-small"
HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"}

def call_huggingface_api(prompt: str):
    response = requests.post(
        HF_API_URL,
        headers=HEADERS,
        json={"inputs": prompt}
    )
    if response.status_code == 200:
        result = response.json()
        return result[0]["generated_text"] if result else "No response from model."
    else:
        return f"Error: {response.status_code} - {response.text}"