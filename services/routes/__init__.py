from .chat import router as chat_router
from .health import router as health_router
from .feedback import router as feedback_router
from .upload import router as upload_router
from .v1 import router as v1_router

__all__ = ["chat_router", "health_router", "feedback_router", "upload_router", "v1_router"]
