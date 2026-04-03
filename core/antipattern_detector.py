"""Anti-pattern detector: rule-based detection of common DS mistakes."""
import ast
from typing import List


class AntiPatternDetector:
    """Detects common Data Science anti-patterns in user-supplied Python code."""

    def __init__(self):
        pass

    def check_code(self, source_code: str) -> List[str]:
        """Scan code for known anti-patterns and return warning strings."""
        warnings = []
        try:
            ast.parse(source_code)
        except SyntaxError:
            return []

        code_str = source_code.lower()
        code_no_space = code_str.replace(" ", "")

        # Data leakage: fit_transform before split
        if "fit_transform" in code_str and "train_test_split" not in code_str:
            if any(pfx in code_str for pfx in ("scaler.fit_transform", "imputer.fit_transform", "encoder.fit_transform")):
                warnings.append(
                    "Data Leakage: Calling fit_transform before train_test_split "
                    "leaks test-set statistics into training. Fit on train only, "
                    "then transform both."
                )

        # Predicting only on train
        if ".predict(x_train)" in code_str and ".predict(x_test)" not in code_str:
            warnings.append(
                "Overfitting risk: Predicting on training data without evaluating "
                "on test data. Always check generalization performance."
            )

        # Blind dropna
        if "dropna()" in code_no_space:
            warnings.append(
                "Indiscriminate dropna() may discard useful rows. "
                "Investigate missing patterns before dropping."
            )

        # No cross-validation
        if ("fit(" in code_str and "score(" in code_str
                and "cross_val" not in code_str and "kfold" not in code_str
                and "gridsearch" not in code_str):
            if "train_test_split" not in code_str:
                warnings.append(
                    "No validation strategy detected. Use train_test_split or "
                    "cross_val_score to get a reliable performance estimate."
                )

        # Accuracy on imbalanced data
        if "accuracy" in code_str and ("imbalance" in code_str or "class_weight" in code_str):
            warnings.append(
                "Accuracy can be misleading on imbalanced data. "
                "Consider F1, precision/recall, or AUC-ROC instead."
            )

        # Fitting on test set
        if ".fit(x_test" in code_no_space or ".fit(xtest" in code_no_space:
            warnings.append(
                "Fitting a model or transformer on the test set defeats "
                "the purpose of held-out evaluation."
            )

        # Using entire dataset for feature selection then splitting
        if "selectkbest" in code_str or "rfe" in code_str:
            if "pipeline" not in code_str and "train_test_split" in code_str:
                if code_str.index("selectkbest" if "selectkbest" in code_str else "rfe") < code_str.index("train_test_split"):
                    warnings.append(
                        "Feature selection before train_test_split uses test "
                        "information to choose features. Wrap selection in a "
                        "Pipeline or apply after splitting."
                    )

        return warnings
