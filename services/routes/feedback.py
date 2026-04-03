from fastapi import APIRouter
from pydantic import BaseModel
import json
import os
from datetime import datetime

router = APIRouter(tags=["feedback"])


class FeedbackRequest(BaseModel):
    query: str
    response: str
    rating: int
    notes: str = ""
    session_id: str = "default"


@router.post("/feedback")
def feedback(payload: FeedbackRequest):
    os.makedirs("storage", exist_ok=True)
    out_path = os.path.join("storage", "feedback.jsonl")
    item = {
        "ts": datetime.utcnow().isoformat(),
        "session_id": payload.session_id,
        "query": payload.query,
        "response": payload.response,
        "rating": payload.rating,
        "notes": payload.notes,
    }
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=True) + "\n")
    return {"saved": True, "rating": payload.rating}
