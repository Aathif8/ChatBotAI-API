import os
from fastapi import APIRouter
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from services.question_service import call_huggingface_api
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
    model="gpt-3.5-turbo",
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

@router.post("/askopenai")
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

@router.post("/askhf")
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
