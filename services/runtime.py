"""Shared runtime singletons for API routes."""
from core.agent import MentorAgent

_agent = None


def get_agent() -> MentorAgent:
    global _agent
    if _agent is None:
        _agent = MentorAgent()
    return _agent
