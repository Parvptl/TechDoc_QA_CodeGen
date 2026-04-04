"""Traceback-aware debugging assistant for common DS errors."""
from dataclasses import dataclass
from typing import Dict, List
import re


@dataclass
class DebugSuggestion:
    error_type: str
    category: str
    line_number: int
    likely_cause: str
    fix: str


ERROR_BANK: List[Dict[str, str]] = [
    {"match": "could not convert string to float", "category": "encoding", "cause": "Categorical text passed into numeric model.", "fix": "Encode categorical columns (OneHotEncoder/OrdinalEncoder) before fitting."},
    {"match": "found input variables with inconsistent numbers of samples", "category": "shape_mismatch", "cause": "X and y lengths differ.", "fix": "Check `len(X)` and `len(y)` after filtering/joins and realign indexes."},
    {"match": "expected 2d array, got 1d array", "category": "shape_mismatch", "cause": "Estimator expected matrix input.", "fix": "Use `x.reshape(-1, 1)` for single feature input."},
    {"match": "settingwithcopywarning", "category": "pandas_assignment", "cause": "Chained indexing created a view copy ambiguity.", "fix": "Use `.loc[row_mask, col] = value` instead of chained assignment."},
    {"match": "keyerror", "category": "column_lookup", "cause": "Column name not present or typo/case mismatch.", "fix": "Print `df.columns` and use exact column names (trim whitespace)." },
    {"match": "module not found", "category": "import", "cause": "Missing dependency.", "fix": "Install dependency or switch to available package in environment."},
    {"match": "nameerror", "category": "variable_scope", "cause": "Variable used before definition.", "fix": "Define variable earlier and verify function/local scope."},
    {"match": "indexerror", "category": "indexing", "cause": "Index out of range after filtering/subsetting.", "fix": "Check lengths and use `.iloc` bounds safely."},
    {"match": "valueerror: could not broadcast", "category": "numpy_shape", "cause": "Array shapes are incompatible.", "fix": "Inspect `.shape` values and align dimensions before assignment/ops."},
    {"match": "memoryerror", "category": "resource", "cause": "Dataset too large for operation.", "fix": "Sample/chunk the dataset or reduce feature size."},
    {"match": "zero division error", "category": "numeric", "cause": "Division by zero occurred.", "fix": "Guard denominator values and handle zero cases explicitly."},
    {"match": "single positional indexer is out-of-bounds", "category": "indexing", "cause": "iloc index outside valid range.", "fix": "Check DataFrame length before positional indexing."},
    {"match": "cannot reindex on an axis with duplicate labels", "category": "indexing", "cause": "Duplicate index labels break reindexing.", "fix": "Reset or deduplicate index before reindex operations."},
    {"match": "columns overlap but no suffix specified", "category": "merge", "cause": "Join introduces duplicate column names.", "fix": "Use suffixes in merge/join to disambiguate overlapping columns."},
    {"match": "mergeerror", "category": "merge", "cause": "Invalid merge keys or ambiguous join configuration.", "fix": "Validate join keys, dtypes, and merge cardinality."},
    {"match": "cannot convert non-finite values", "category": "missing_values", "cause": "NaN/inf values encountered during cast.", "fix": "Replace or impute NaN/inf before conversion."},
    {"match": "notfittederror", "category": "model_lifecycle", "cause": "Estimator used before fitting.", "fix": "Call `.fit()` before `.predict()` or `.transform()`."},
    {"match": "solver failed to converge", "category": "optimization", "cause": "Model optimization did not converge.", "fix": "Scale features, increase max_iter, or adjust regularization."},
    {"match": "input contains nan", "category": "missing_values", "cause": "Estimator cannot handle missing values.", "fix": "Impute missing values or use estimators that support NaN."},
    {"match": "expected sequence or array-like", "category": "input_type", "cause": "Estimator got unsupported input type.", "fix": "Pass numpy array/DataFrame and verify shape."},
    {"match": "buffer dtype mismatch", "category": "dtype", "cause": "Internal operation expected different dtype.", "fix": "Cast arrays explicitly (e.g., float64)." },
    {"match": "could not interpret value", "category": "visualization", "cause": "Plot references a column that does not exist.", "fix": "Check DataFrame column names in plotting call."},
    {"match": "singular matrix", "category": "linear_algebra", "cause": "Matrix inversion failed due to multicollinearity.", "fix": "Remove collinear features or regularize model."},
    {"match": "x has", "category": "feature_count", "cause": "Feature count mismatch between train and inference.", "fix": "Ensure identical preprocessing pipeline at train/inference."},
    {"match": "n_splits", "category": "cross_validation", "cause": "CV folds exceed available samples per class.", "fix": "Reduce number of folds or collect more data."},
    {"match": "at least one array or dtype is required", "category": "empty_input", "cause": "Model got empty or None input.", "fix": "Validate non-empty X and y before fitting."},
    {"match": "object of type 'float' has no len", "category": "type_error", "cause": "Scalar used where sequence expected.", "fix": "Wrap scalar into list/array if sequence required."},
    {"match": "unexpected keyword argument", "category": "api_mismatch", "cause": "Wrong parameter for function/estimator version.", "fix": "Check library version and parameter names."},
    {"match": "positional argument", "category": "api_mismatch", "cause": "Function called with wrong number/order of args.", "fix": "Match call signature from docs."},
    {"match": "attributeerror", "category": "attribute_access", "cause": "Object does not expose requested attribute/method.", "fix": "Check object type and available attributes via dir()."},
    {"match": "typeerror", "category": "type_error", "cause": "Operation performed on incompatible types.", "fix": "Inspect variable types and cast appropriately."},
    {"match": "filenotfounderror", "category": "io", "cause": "Input file path not found.", "fix": "Use correct relative/absolute path and verify file exists."},
    {"match": "permissionerror", "category": "io", "cause": "File access denied.", "fix": "Write to allowed directory and close open handles."},
    {"match": "unicode decode error", "category": "encoding", "cause": "Wrong file encoding.", "fix": "Pass correct encoding in read function (e.g., utf-8, latin1)."},
    {"match": "parsererror", "category": "csv_parse", "cause": "CSV delimiter/quoting mismatch.", "fix": "Set proper delimiter, quoting, or engine in read_csv."},
    {"match": "valueerror: y contains previously unseen labels", "category": "encoding", "cause": "LabelEncoder saw unseen categories at inference.", "fix": "Use fit-on-train only and handle unseen categories safely."},
    {"match": "precision is ill-defined", "category": "metric_warning", "cause": "No predicted samples for a class.", "fix": "Inspect class imbalance and threshold/model behavior."},
    {"match": "convergencewarning", "category": "optimization", "cause": "Estimator convergence warning.", "fix": "Scale data or increase iteration limits."},
    {"match": "cannot do a non-empty take from an empty axes", "category": "empty_data", "cause": "Operation called on empty data subset.", "fix": "Validate filter result before aggregation/indexing."},
]


class DebugAssistant:
    """Parses traceback strings and proposes targeted fixes."""

    def __init__(self, error_bank: List[Dict[str, str]] = None):
        self.error_bank = error_bank or ERROR_BANK

    @staticmethod
    def looks_like_traceback(text: str) -> bool:
        t = (text or "").lower()
        return ("traceback" in t) or ("error:" in t) or ("exception" in t)

    @staticmethod
    def extract_error_type(text: str) -> str:
        lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
        for ln in reversed(lines):
            m = re.match(r"([A-Za-z_]*Error)\s*:", ln)
            if m:
                return m.group(1)
        return "UnknownError"

    @staticmethod
    def extract_line_number(text: str) -> int:
        lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
        for ln in lines:
            if ", line " in ln:
                try:
                    after = ln.split(", line ", 1)[1]
                    num = ""
                    for ch in after:
                        if ch.isdigit():
                            num += ch
                        else:
                            break
                    if num:
                        return int(num)
                except Exception:
                    continue
        return -1

    def suggest(self, traceback_text: str) -> DebugSuggestion:
        text = (traceback_text or "").lower()
        error_type = self.extract_error_type(traceback_text)
        line_number = self.extract_line_number(traceback_text)
        for entry in self.error_bank:
            if entry["match"] in text:
                return DebugSuggestion(
                    error_type=error_type,
                    category=entry["category"],
                    line_number=line_number,
                    likely_cause=entry["cause"],
                    fix=entry["fix"],
                )
        return DebugSuggestion(
            error_type=error_type,
            category="general",
            line_number=line_number,
            likely_cause="The traceback needs step-by-step inspection.",
            fix="Check the failing line, print shapes/types, and validate column names before retrying.",
        )
