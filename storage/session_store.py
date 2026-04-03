import os
import json
from typing import Dict, Any

class SessionStore:
    def __init__(self, filename="storage/sessions.json"):
        self.filename = filename
        self.sessions = {}
        os.makedirs(os.path.dirname(self.filename), exist_ok=True)
        if os.path.exists(self.filename):
            try:
                with open(self.filename, "r", encoding="utf-8") as f:
                    self.sessions = json.load(f)
            except (json.JSONDecodeError, OSError):
                self.sessions = {}
                
    def save(self, session_id: str, data: Dict[str, Any]):
        self.sessions[session_id] = data
        with open(self.filename, "w", encoding="utf-8") as f:
            json.dump(self.sessions, f, ensure_ascii=True, indent=2)
            
    def load(self, session_id: str) -> Dict[str, Any]:
        return self.sessions.get(session_id, {})
