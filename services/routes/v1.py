"""Versioned API v1 routes."""
from datetime import datetime
import os
from typing import Optional, Dict, Any, List

from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel

from services.runtime import get_agent
from core.antipattern_detector import AntiPatternDetector

router = APIRouter(prefix="/api/v1", tags=["v1"])
_detector = AntiPatternDetector()


class ChatV1Request(BaseModel):
    query: str
    session_id: str = "default"
    code: str = ""
    stage_hint: Optional[int] = None
    dataset_context: Optional[str] = None


class QuizRequest(BaseModel):
    session_id: str = "default"
    stage: Optional[int] = None
    difficulty: Optional[str] = None


class NotebookReviewRequest(BaseModel):
    notebook_json: Dict[str, Any]

class CheckpointSubmitRequest(BaseModel):
    session_id: str = "default"
    stage_num: int
    evidence: str = ""


class QuizGradeRequest(BaseModel):
    session_id: str = "default"
    stage: Optional[int] = None
    answers: List[Dict[str, str]] = []


@router.post("/chat")
def chat_v1(payload: ChatV1Request):
    try:
        return get_agent().process_sync(
            query=payload.query,
            provided_code=payload.code,
            session_id=payload.session_id,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/upload")
async def upload_v1(file: UploadFile = File(...), session_id: str = "default"):
    os.makedirs("storage/uploads", exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
    dataset_id = f"ds_{timestamp}"
    out_path = os.path.join("storage", "uploads", f"{dataset_id}_{file.filename}")

    content = await file.read()
    with open(out_path, "wb") as out_file:
        out_file.write(content)

    result: Dict[str, Any] = {
        "dataset_id": dataset_id,
        "filename": file.filename,
        "bytes": len(content),
        "stored_as": out_path,
    }
    ext = (file.filename or "").rsplit(".", 1)[-1].lower()
    if ext in ("csv", "xlsx", "xls"):
        try:
            profile = get_agent().upload_dataset(content, file.filename)
            result["profile_summary"] = {
                "rows": profile.get("rows", 0),
                "columns": profile.get("columns", 0),
                "target_guess": profile.get("target_guess", ""),
            }
            result["project_plan"] = get_agent().get_project_plan()
            result["suggested_questions"] = [
                "What should be my baseline metric for this dataset?",
                "Can you guide me through EDA checkpoints first?",
                "What preprocessing steps should I prioritize next?",
            ]
        except Exception as exc:
            result["profile_error"] = str(exc)
    return result


@router.get("/session/{session_id}/progress")
def session_progress(session_id: str):
    try:
        return get_agent().get_progress(session_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/session/{session_id}/dashboard")
def session_dashboard(session_id: str):
    try:
        return get_agent().get_learning_dashboard(session_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/session/{session_id}/report")
def session_report(session_id: str, format: str = "markdown"):
    try:
        return get_agent().export_learning_report(session_id=session_id, fmt=format)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/quiz/generate")
def generate_quiz(payload: QuizRequest):
    agent = get_agent()
    return agent.generate_quiz(
        session_id=payload.session_id,
        stage=payload.stage,
        difficulty=payload.difficulty,
    )


@router.post("/quiz/grade")
def grade_quiz(payload: QuizGradeRequest):
    agent = get_agent()
    return agent.grade_quiz(
        session_id=payload.session_id,
        answers=payload.answers,
        stage=payload.stage,
    )


@router.post("/notebook/review")
def notebook_review(payload: NotebookReviewRequest):
    cells = payload.notebook_json.get("cells", [])
    code_cells = [c for c in cells if c.get("cell_type") == "code"]
    markdown_cells = [c for c in cells if c.get("cell_type") == "markdown"]
    code_text = "\n\n".join(_cell_source(c) for c in code_cells)
    anti_warnings = _detector.check_code(code_text) if code_text.strip() else []

    suggestions = []
    if len(code_cells) < 3:
        suggestions.append("Add more executable analysis cells across EDA, preprocessing, and modeling.")
    if len(markdown_cells) < 2:
        suggestions.append("Add markdown explanations of assumptions, metrics, and conclusions.")
    if anti_warnings:
        suggestions.append("Notebook contains code anti-patterns that should be fixed before final submission.")
    stage_gaps = _infer_stage_gaps(code_text)
    for gap in stage_gaps:
        suggestions.append(gap)
    if not suggestions:
        suggestions.append("Good structure. Add explicit leakage checks and final error analysis section.")

    penalty = min(20, 4 * len(anti_warnings))
    score = max(0, min(100, 60 + (len(code_cells) * 5) + (len(markdown_cells) * 4) - penalty))
    return {
        "review": suggestions,
        "score": score,
        "suggestions": suggestions,
        "antipattern_warnings": anti_warnings,
        "stage_gaps": stage_gaps,
    }


@router.post("/project/checkpoint")
def project_checkpoint_submit(payload: CheckpointSubmitRequest):
    agent = get_agent()
    return agent.submit_project_checkpoint(
        session_id=payload.session_id,
        stage_num=payload.stage_num,
        evidence=payload.evidence,
    )


def _next_focus_stage(stage_mastery: Dict[Any, Any]) -> int:
    if not stage_mastery:
        return 1
    parsed = []
    for k, v in stage_mastery.items():
        try:
            parsed.append((int(k), float(v)))
        except (TypeError, ValueError):
            continue
    if not parsed:
        return 1
    parsed.sort(key=lambda x: x[1])
    return int(parsed[0][0])


def _cell_source(cell: Dict[str, Any]) -> str:
    src = cell.get("source", "")
    if isinstance(src, list):
        return "".join(str(x) for x in src)
    return str(src)


def _infer_stage_gaps(code_text: str) -> List[str]:
    lower = (code_text or "").lower()
    gaps = []
    checks = [
        ("train_test_split", "Missing explicit train/test split step."),
        ("fillna", "Missing visible missing-value handling step (imputation)."),
        ("fit(", "Missing model training step."),
        ("score(", "Missing explicit evaluation call (`score` or metric function)."),
    ]
    for token, msg in checks:
        if token not in lower:
            gaps.append(msg)
    return gaps
