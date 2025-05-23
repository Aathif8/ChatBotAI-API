import requests
import os
import openai
import json
from dotenv import load_dotenv
from services.upload_service import get_chroma_collections

load_dotenv()

HF_API_KEY = os.getenv("HF_API_KEY")
OPEN_ROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
HF_TRANSCRIBE_URL = "https://api-inference.huggingface.co/models/openai/whisper-small"
HEADERS = {
    "Authorization": f"Bearer {OPEN_ROUTER_API_KEY}",
    "Content-Type": "application/json"
    }


# Embedding the Query to retrieve relevant data from ChromaDB
def retrieve_relevant_data(query):
    query_embedding_response = openai.embeddings.create(
        model="text-embedding-ada-002",
        input=[query]
    )
    query_embedding = query_embedding_response.data[0].embedding

    results = get_chroma_collections().query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    relevant_texts = " ".join(results["documents"][0] if results["documents"] else "No relevant data found")
    return relevant_texts

def OpenRouterAPI(prompt: str):
    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers=HEADERS,
        data=json.dumps({
            "model": "meta-llama/llama-4-maverick:free",
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
        })
    )
    if response.status_code == 200:
        result = response.json()
        return result["choices"][0]["message"]["content"] if result else "No response from model."
    else:
        return f"Error: {response.status_code} - {response.text}"