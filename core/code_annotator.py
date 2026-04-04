"""Line-by-line explanation helper for generated Python code."""
from typing import List


class CodeAnnotator:
    """Annotates common DS code patterns with educational comments."""

    _PATTERNS = [
        ("train_test_split(", "Split data into train and test partitions."),
        ("fit(", "Learn parameters from training data."),
        ("predict(", "Generate predictions from fitted model."),
        ("read_csv(", "Load tabular data into a DataFrame."),
        ("fillna(", "Handle missing values using a chosen strategy."),
        ("dropna(", "Remove rows/columns with missing values."),
        ("StandardScaler(", "Scale numeric features to zero mean and unit variance."),
        ("OneHotEncoder(", "Convert categorical values into binary indicator columns."),
        ("LabelEncoder(", "Convert labels/categories into integer-encoded values."),
        ("RandomForest", "Tree ensemble model; robust for many tabular tasks."),
        ("cross_val_score(", "Estimate model quality with cross-validation."),
        ("classification_report(", "Summarize precision, recall, and F1 metrics."),
        ("confusion_matrix(", "Inspect class-wise prediction errors."),
        ("roc_auc_score(", "Compute threshold-independent classifier quality."),
        ("plt.", "Create or customize a visualization."),
    ]

    def annotate(self, code: str) -> str:
        lines = (code or "").splitlines()
        if not lines:
            return ""
        out: List[str] = []
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                out.append(line)
                continue

            comment = self._line_comment(stripped)
            if comment:
                out.append(f"{line}  # {comment}")
            else:
                out.append(line)
        return "\n".join(out)

    def _line_comment(self, stripped_line: str) -> str:
        lower = stripped_line.lower()
        for token, explanation in self._PATTERNS:
            if token.lower() in lower:
                return explanation
        if "=" in stripped_line and "==" not in stripped_line and not stripped_line.startswith("for "):
            return "Assign result to a variable for later steps."
        return ""
