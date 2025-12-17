from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ai import get_ai_response

app = FastAPI(title="Python Tutor API")

class UserQuery(BaseModel):
    question: str

@app.get("/")
def read_root():
    return {"message": "🐍 Python Tutor API is running!"}

@app.post("/ask")
def ask_question(query: UserQuery):
    try:
        answer = get_ai_response(query.question)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))