from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
import requests
import openai
import io

load_dotenv()

app = FastAPI()

# Allow frontend origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # React Vite default
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HF_API_KEY = os.getenv("HF_API_KEY")  # store your key in .env
HF_API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-base"
HF_TRANSCRIBE_URL = "https://api-inference.huggingface.co/models/openai/whisper-small"
HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"}

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# Mock user data
user_data = {
    "account_balance": "$2,450.75",
    "loan_approved": True
}

class QuestionRequest(BaseModel):
    question: str
    history: list[str] = []

# Functioon to transcribe audio
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

@app.post("/askopenai")
async def ask_bot(req: QuestionRequest):
    question = req.question
    history = req.history

    if question.lower() == "what’s my account balance?":
        return {"answer": f"Your current account balance is {user_data['account_balance']}"}
    
    elif question.lower() == "is my loan approved?":
        approved = user_data["loan_approved"]
        return {"answer": "Yes, your loan has been approved!" if approved else "Your loan is still under review."}

    elif question.lower() == "how can i apply for a loan?":
        system_msg = (
            "You are a helpful banking assistant. If a user asks about applying for a loan, "
            "ask clarifying questions such as the loan type and amount."
        )
    elif question.lower() == "i saw a suspicious transaction.":
        system_msg = (
            "You are a banking assistant. Ask for transaction date, amount, and merchant."
        )
    else:
        system_msg = "You are a helpful banking assistant."

    messages = [HumanMessage(content=system_msg), HumanMessage(content=question)]
    messages += [HumanMessage(content=msg) for msg in history]
    response = llm(messages)

    return {"answer": response.content}

@app.post("/askhf")
async def ask_bot(req: QuestionRequest):
    question = req.question.lower()

    # Handle predefined logic
    if question == "what’s my account balance?":
        return {"answer": f"Your current account balance is {user_data['account_balance']}"}
    elif question == "is my loan approved?":
        return {
            "answer": "Yes, your loan has been approved!" if user_data["loan_approved"]
            else "Your loan is still under review."
        }

    # Otherwise, ask the hosted model
    prompt = f"You are a helpful banking assistant.\nUser: {req.question}"
    answer = call_huggingface_api(prompt)
    return {"answer": answer}

@app.post("/transcribeopenai")
async def process_audio(file: UploadFile = File(...)):
    audio_bytes = await file.read()

    transcription = transcribe_audio(audio_bytes)
    return {"openaitranscription": transcription}


@app.post("/transcribehf")
async def process_audio(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    transcription = TranscribeHF(audio_bytes)
    return {"hftranscription": transcription}
