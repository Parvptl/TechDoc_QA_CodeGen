from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services.runtime import get_agent

router = APIRouter(tags=["chat"])


class QueryRequest(BaseModel):
    query: str
    code: str = ""
    session_id: str = "default"


@router.post("/chat")
def chat(request: QueryRequest):
    try:
        return get_agent().process_sync(request.query, request.code, request.session_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
