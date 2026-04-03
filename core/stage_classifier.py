import re
from typing import Dict, List, Optional, Tuple

class StageClassifier:
    """Classifies user queries into one of the 7 Data Science pipeline stages."""
    
    STAGE_KEYWORDS = {
        1: ["problem", "goal", "objective", "target", "metric", "baseline"],
        2: ["load", "read", "import", "csv", "json", "sql", "shape", "dtype", "fetching"],
        3: ["explore", "distribution", "correlation", "outlier", "missing", "pattern", "eda", "plot", "histogram"],
        4: ["preprocess", "imput", "cap", "scale", "encod", "clean", "drop", "fill"],
        5: ["feature", "engineer", "polynomial", "pca", "extract", "datetime", "creation"],
        6: ["model", "train", "fit", "random forest", "xgboost", "predict", "cross-validation", "tune", "hyperparameter"],
        7: ["eval", "auc", "roc", "confusion matrix", "learning curve", "shap", "accuracy", "f1", "score"]
    }
    
    STAGE_NAMES = {
        1: "Problem Understanding",
        2: "Data Loading",
        3: "Exploratory Data Analysis",
        4: "Preprocessing",
        5: "Feature Engineering",
        6: "Modeling",
        7: "Evaluation"
    }

    def __init__(self):
        # We can implement TF-IDF + SVM here, but for zero-setup, we use keyword heuristic
        # backed by a regex matcher. It operates identically from an API perspective.
        pass

    def classify(self, query: str) -> Tuple[int, float]:
        """Returns the predicted stage (1-7) and confidence (0.0-1.0)."""
        query_lower = query.lower()
        scores = {stage: 0 for stage in self.STAGE_NAMES}
        
        for stage, keywords in self.STAGE_KEYWORDS.items():
            for kw in keywords:
                if re.search(r'\b' + kw + r'\b', query_lower):
                    scores[stage] += 1
                elif kw in query_lower:
                    scores[stage] += 0.5  # partial match
        
        max_score = max(scores.values())
        if max_score == 0:
            return 1, 0.1  # default to stage 1 if completely unknown
            
        best_stage = max(scores, key=scores.get)
        total_score = sum(scores.values()) + 1e-5
        confidence = min(0.95, (scores[best_stage] / total_score) + 0.1 * scores[best_stage])
        
        return best_stage, confidence

    def get_stage_name(self, stage: int) -> str:
        return self.STAGE_NAMES.get(stage, "Unknown Stage")
