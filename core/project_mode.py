"""Minimal project-based learning mode."""
from typing import Dict, List, Iterable


STAGE_NAMES = {
    1: "Problem Understanding",
    2: "Data Loading",
    3: "Exploratory Data Analysis",
    4: "Preprocessing",
    5: "Feature Engineering",
    6: "Modeling",
    7: "Evaluation",
}


class ProjectModeEngine:
    """Builds a guided DS project plan with checkpoints."""

    def generate_plan(
        self,
        dataset_profile: Dict,
        stage_skills: Dict[int, float],
        completed_stages: Iterable[int] = None,
    ) -> Dict:
        filename = dataset_profile.get("filename", "uploaded dataset")
        rows = dataset_profile.get("rows", 0)
        cols = dataset_profile.get("columns", 0)
        target = dataset_profile.get("target_guess") or "target column"

        title = f"Predict outcomes from {filename}"
        brief = (
            f"Build an end-to-end model using {filename} "
            f"({rows} rows, {cols} columns) and evaluate it rigorously."
        )

        done = set(int(x) for x in (completed_stages or []))
        next_required = self.next_required_stage(done)
        checkpoints: List[Dict] = []
        for stage in range(1, 8):
            mastery = float(stage_skills.get(stage, 0.5))
            status = "ready" if mastery >= 0.45 else "focus"
            unlocked = stage <= next_required
            checkpoints.append(
                {
                    "stage_num": stage,
                    "stage_name": STAGE_NAMES[stage],
                    "mastery": round(mastery, 2),
                    "status": status,
                    "unlocked": unlocked,
                    "checkpoint": self._checkpoint_text(stage, target),
                }
            )

        return {
            "mode": "project",
            "title": title,
            "brief": brief,
            "dataset_summary": {"filename": filename, "rows": rows, "columns": cols, "target_guess": target},
            "checkpoints": checkpoints,
            "next_required_stage": next_required,
        }

    @staticmethod
    def next_required_stage(completed_stages: Iterable[int]) -> int:
        done = set(int(x) for x in (completed_stages or []))
        for stage in range(1, 8):
            if stage not in done:
                return stage
        return 7

    def gate_stage_jump(self, predicted_stage: int, completed_stages: Iterable[int]) -> Dict:
        next_required = self.next_required_stage(completed_stages)
        predicted_stage = int(predicted_stage)
        blocked = predicted_stage > next_required
        if not blocked:
            return {"blocked": False, "next_required_stage": next_required, "message": ""}
        return {
            "blocked": True,
            "next_required_stage": next_required,
            "message": (
                f"Project mode checkpoint: complete Stage {next_required} "
                f"before jumping to Stage {predicted_stage}."
            ),
        }

    @staticmethod
    def _checkpoint_text(stage: int, target: str) -> str:
        mapping = {
            1: f"Define objective, success metric, and baseline for predicting `{target}`.",
            2: "Load dataset, verify dtypes, shape, and basic data quality.",
            3: "Run EDA: distributions, missingness, and key correlations.",
            4: "Preprocess: missing value handling, encoding, scaling decisions.",
            5: "Engineer or select useful features with rationale.",
            6: "Train at least one baseline and one stronger model.",
            7: "Evaluate with task-appropriate metrics and error analysis.",
        }
        return mapping.get(stage, "Complete stage tasks.")
