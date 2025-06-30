import os
from fastapi import APIRouter
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from services.question_service import OpenRouterAPI, retrieve_relevant_data

load_dotenv()

router = APIRouter()

class QuestionRequest(BaseModel):
    question: str
    history: list[str] = []

# Mock user data
user_data = {
    "account_balance": "$2,450.75",
    "loan_approved": True
}

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

@router.post("/askopenai")
async def ask_bot(req: QuestionRequest):
    query = req.question.lower()
    context = retrieve_relevant_data(query)

    prompt = (
        "கீழே வழங்கப்பட்ட உள்ளடக்கத்தைப் பயன்படுத்தி கேள்விக்குப் பதில் அளிக்கவும்.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{query}\n\n"
        "உங்கள் பதில் தமிழ் மொழியில் மட்டுமே இருக்க வேண்டும்."
    )

    response = llm([HumanMessage(content=prompt)])

    return {"answer": response.content}

@router.post("/askhf")
async def ask_bot(req: QuestionRequest):
    query = req.question.lower()
    context = retrieve_relevant_data(query)

    prompt = (
        "கீழே வழங்கப்பட்ட உள்ளடக்கத்தைப் பயன்படுத்தி கேள்விக்குப் பதில் அளிக்கவும்.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{query}\n\n"
        "உங்கள் பதில் தமிழ் மொழியில் மட்டுமே இருக்க வேண்டும்."
    )

    answer = OpenRouterAPI(prompt)
    return {"answer": answer}
