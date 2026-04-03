from typing import List, Dict, Any

class UserProfile:
    """Tracks generic skill level and error history for adaptive difficulty."""
    def __init__(self):
        self.points = 0
        self.stage_skills = {i: 0.5 for i in range(1, 8)} # Bayseian skill roughly 0.0 - 1.0
        self.total_queries = 0

class SessionMemory:
    """Stores the context of a multi-turn conversation."""
    def __init__(self):
        self.history: List[Dict[str, str]] = []
        self.profile = UserProfile()
        self.active_context = {}

    def add_turn(self, role: str, content: str):
        self.history.append({"role": role, "content": content})
        if role == "user":
            self.profile.total_queries += 1
            
    def get_recent_context(self, turns: int = 3) -> str:
        recent = self.history[-turns*2:] if self.history else []
        return "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in recent])
        
    def update_context(self, key: str, value: Any):
        self.active_context[key] = value
