from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from routes.question import router as question_router
from routes.transcribe import router as transcribe_router
from routes.upload import router as upload_router

load_dotenv()

app = FastAPI()

# Allow frontend origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload_router, prefix="/api")
app.include_router(question_router, prefix="/api")
app.include_router(transcribe_router, prefix="/api")