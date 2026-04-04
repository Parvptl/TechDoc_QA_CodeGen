"""DS Mentor Pro -- FastAPI application with static frontend."""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from services.routes import chat_router, health_router, feedback_router, upload_router, v1_router

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = FastAPI(title="DS Mentor Pro API", version="1.0.0")

app.include_router(health_router)
app.include_router(chat_router)
app.include_router(upload_router)
app.include_router(feedback_router)
app.include_router(v1_router)

app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")


@app.get("/")
async def root():
    return FileResponse(os.path.join(BASE_DIR, "templates", "index.html"))
