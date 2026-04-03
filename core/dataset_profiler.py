"""
Dataset Profiler: Parses uploaded CSV/Excel files, detects schema,
computes summary statistics, and provides context for dataset-aware responses.
"""
import io
import json
from typing import Dict, Any, Optional


class DatasetProfiler:
    """Analyses an uploaded dataset and produces a structured profile."""

    def __init__(self):
        self._profile: Optional[Dict[str, Any]] = None
        self._df = None

    @property
    def profile(self) -> Optional[Dict[str, Any]]:
        return self._profile

    @property
    def dataframe(self):
        return self._df

    @property
    def is_loaded(self) -> bool:
        return self._profile is not None

    def load_csv(self, file_bytes: bytes, filename: str = "dataset.csv") -> Dict[str, Any]:
        """Load CSV from raw bytes and return a profile dict."""
        import pandas as pd
        self._df = pd.read_csv(io.BytesIO(file_bytes))
        self._profile = self._build_profile(filename)
        return self._profile

    def load_excel(self, file_bytes: bytes, filename: str = "dataset.xlsx") -> Dict[str, Any]:
        """Load Excel from raw bytes and return a profile dict."""
        import pandas as pd
        self._df = pd.read_excel(io.BytesIO(file_bytes))
        self._profile = self._build_profile(filename)
        return self._profile

    def load_from_bytes(self, file_bytes: bytes, filename: str) -> Dict[str, Any]:
        """Auto-detect format from filename extension."""
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if ext in ("xlsx", "xls"):
            return self.load_excel(file_bytes, filename)
        return self.load_csv(file_bytes, filename)

    def _build_profile(self, filename: str) -> Dict[str, Any]:
        """Build a structured profile from the loaded DataFrame."""
        df = self._df
        rows, cols = df.shape

        column_info = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            missing = int(df[col].isnull().sum())
            missing_pct = round(missing / rows * 100, 1) if rows > 0 else 0.0
            unique = int(df[col].nunique())

            info = {
                "name": col,
                "dtype": dtype,
                "missing": missing,
                "missing_pct": missing_pct,
                "unique": unique,
            }

            if dtype in ("float64", "int64", "float32", "int32"):
                info["mean"] = round(float(df[col].mean()), 3) if missing < rows else None
                info["std"] = round(float(df[col].std()), 3) if missing < rows else None
                info["min"] = round(float(df[col].min()), 3) if missing < rows else None
                info["max"] = round(float(df[col].max()), 3) if missing < rows else None
            elif dtype == "object" or dtype == "category":
                top_values = df[col].value_counts().head(5).to_dict()
                info["top_values"] = {str(k): int(v) for k, v in top_values.items()}

            column_info.append(info)

        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        target_guess = None
        for candidate in ["target", "label", "class", "survived", "y"]:
            matches = [c for c in df.columns if candidate in c.lower()]
            if matches:
                target_guess = matches[0]
                break

        return {
            "filename": filename,
            "rows": rows,
            "columns": cols,
            "column_names": df.columns.tolist(),
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "column_info": column_info,
            "total_missing": int(df.isnull().sum().sum()),
            "total_missing_pct": round(df.isnull().sum().sum() / (rows * cols) * 100, 1) if rows * cols > 0 else 0.0,
            "target_guess": target_guess,
        }

    def get_context_string(self) -> str:
        """Return a concise text summary for injection into the generator."""
        if not self._profile:
            return ""

        p = self._profile
        lines = [
            f"Uploaded dataset: {p['filename']} ({p['rows']} rows, {p['columns']} columns)",
            f"Numeric columns: {', '.join(p['numeric_columns'][:10])}",
            f"Categorical columns: {', '.join(p['categorical_columns'][:10])}",
        ]

        if p["total_missing"] > 0:
            lines.append(f"Missing values: {p['total_missing']} ({p['total_missing_pct']}% of all cells)")

        high_missing = [c for c in p["column_info"] if c["missing_pct"] > 20]
        if high_missing:
            cols = ", ".join(f"{c['name']} ({c['missing_pct']}%)" for c in high_missing[:5])
            lines.append(f"Columns with >20% missing: {cols}")

        if p["target_guess"]:
            lines.append(f"Likely target column: {p['target_guess']}")

        return "\n".join(lines)

    def get_column_list(self) -> list:
        """Return column names for code generation context."""
        if not self._profile:
            return []
        return self._profile["column_names"]
