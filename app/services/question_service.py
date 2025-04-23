import requests
import os
import openai
from dotenv import load_dotenv
from llama_cpp import Llama
from services.upload_service import get_chroma_collections

load_dotenv()

HF_API_KEY = os.getenv("HF_API_KEY")
HF_API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-base"
HF_TRANSCRIBE_URL = "https://api-inference.huggingface.co/models/openai/whisper-small"
HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"}

llm_model = Llama(model_path="./models/models--TheBloke--TinyLlama-1.1B-Chat-v1.0-GGUF/snapshots/52e7645ba7c309695bec7ac98f4f005b139cf465/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf", n_ctx=1024, n_batch=64, temperature=0.1, top_p=0.95, repeat_penalty=1.2)

# llm = Llama.from_pretrained(
# 	repo_id="TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
# 	filename="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
#     temperature=0.1,
#     top_p=0.95,
#     repeat_penalty=1.2,
#     n_ctx=1024,
#     n_batch=64,
# )


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
    
def call_llama(prompt: str):
    response = llm_model(prompt, max_tokens=1000)
    return response["choices"][0]["text"] if "choices" in response else "No response from model."