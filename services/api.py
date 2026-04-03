import sys
import os
from fastapi import FastAPI

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services.routes import chat_router, health_router, feedback_router, upload_router

app = FastAPI(title="DS Mentor Pro API", version="1.0.0")
app.include_router(health_router)
app.include_router(chat_router)
app.include_router(upload_router)
app.include_router(feedback_router)
