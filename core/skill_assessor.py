import random

class SkillAssessor:
    """Adapts difficulty level based on user interaction history."""
    
    def __init__(self, memory):
        self.memory = memory

    def update_skill(self, stage: int, success: bool):
        profile = self.memory.profile
        if stage not in profile.stage_skills:
            return
            
        current = profile.stage_skills[stage]
        if success:
            profile.stage_skills[stage] = min(1.0, current + 0.1)
        else:
            profile.stage_skills[stage] = max(0.0, current - 0.15)
            
    def get_difficulty_string(self, stage: int) -> str:
        skill = self.memory.profile.stage_skills.get(stage, 0.5)
        if skill < 0.3:
            return "beginner"
        elif skill < 0.7:
            return "intermediate"
        else:
            return "advanced"
