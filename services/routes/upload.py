"""Upload endpoint with dataset profiling."""
from fastapi import APIRouter, UploadFile, File
import os
from datetime import datetime

router = APIRouter(tags=["upload"])

_profiler = None


def _get_profiler():
    """Lazy-init the dataset profiler to avoid circular imports."""
    global _profiler
    if _profiler is None:
        from core.dataset_profiler import DatasetProfiler
        _profiler = DatasetProfiler()
    return _profiler


@router.post("/upload")
async def upload(file: UploadFile = File(...)):
    os.makedirs("storage/uploads", exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
    safe_name = f"{timestamp}_{file.filename}"
    out_path = os.path.join("storage", "uploads", safe_name)

    content = await file.read()
    with open(out_path, "wb") as out_file:
        out_file.write(content)

    result = {
        "filename": file.filename,
        "content_type": file.content_type,
        "bytes": len(content),
        "stored_as": out_path,
    }

    ext = (file.filename or "").rsplit(".", 1)[-1].lower()
    if ext in ("csv", "xlsx", "xls"):
        try:
            profiler = _get_profiler()
            profile = profiler.load_from_bytes(content, file.filename)
            result["profile"] = profile
        except Exception as exc:
            result["profile_error"] = str(exc)

    return result
