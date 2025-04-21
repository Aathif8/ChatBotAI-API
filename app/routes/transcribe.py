from fastapi import APIRouter, UploadFile, File
from services.transcribe_service import transcribe_audio, TranscribeHF

router = APIRouter()

@router.post("/transcribeopenai")
async def process_audio(file: UploadFile = File(...)):
    audio_bytes = await file.read()

    transcription = transcribe_audio(audio_bytes)
    return {"openaitranscription": transcription}


@router.post("/transcribehf")
async def process_audio(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    transcription = TranscribeHF(audio_bytes)
    return {"hftranscription": transcription}