from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from core.agent import MentorAgent

router = APIRouter(tags=["chat"])
agent = MentorAgent()


class QueryRequest(BaseModel):
    query: str
    code: str = ""
    session_id: str = "default"


@router.post("/chat")
def chat(request: QueryRequest):
    try:
        return agent.process_sync(request.query, request.code, request.session_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
