"""Pipeline Tracker: 7-stage workflow state machine with skip detection
and intelligent next-step suggestions."""
from typing import List, Set


class PipelineTracker:
    """Tracks user progress through the 7 data science stages."""

    PREREQUISITES = {
        1: [],
        2: [1],
        3: [2],
        4: [2, 3],
        5: [2, 4],
        6: [2, 3, 4],
        7: [6],
    }

    STAGE_NAMES = {
        1: "Problem Understanding",
        2: "Data Loading",
        3: "Exploratory Data Analysis",
        4: "Preprocessing",
        5: "Feature Engineering",
        6: "Modeling",
        7: "Evaluation",
    }

    NEXT_STEP_HINTS = {
        1: "Consider loading your dataset next so you can inspect its structure.",
        2: "A good next step is exploratory data analysis -- check distributions, correlations, and missing values.",
        3: "Now that you have explored the data, handle missing values, outliers, and encoding in preprocessing.",
        4: "With clean data you can engineer new features or proceed to modeling.",
        5: "Your features are ready. Try training a baseline model to see initial results.",
        6: "After training, evaluate your model with appropriate metrics on held-out data.",
        7: "You have completed the full pipeline. Consider iterating on earlier stages to improve results.",
    }

    def __init__(self):
        self.completed_stages: Set[int] = set()

    def mark_completed(self, stage: int):
        """Record that a stage has been visited."""
        if 1 <= stage <= 7:
            self.completed_stages.add(stage)

    def check_prerequisites(self, target_stage: int) -> List[str]:
        """Return warning strings for any unmet prerequisites."""
        warnings = []
        if target_stage not in self.PREREQUISITES:
            return warnings

        for req in self.PREREQUISITES[target_stage]:
            if req not in self.completed_stages:
                warnings.append(
                    f"You are jumping to {self.STAGE_NAMES[target_stage]} "
                    f"without completing {self.STAGE_NAMES[req]}."
                )
        return warnings

    def suggest_next_step(self) -> str:
        """Return a brief suggestion for the logical next stage."""
        for stage in range(1, 8):
            if stage not in self.completed_stages:
                return self.NEXT_STEP_HINTS.get(stage, "")
        return self.NEXT_STEP_HINTS[7]

    def progress_fraction(self) -> float:
        """Return completion as a float 0.0-1.0."""
        return len(self.completed_stages) / 7.0
