"""Tests for core.dataset_profiler.DatasetProfiler."""
import io
import pytest
from core.dataset_profiler import DatasetProfiler


def _csv_bytes(text: str) -> bytes:
    return text.encode("utf-8")


SAMPLE_CSV = _csv_bytes(
    "age,income,city,target\n"
    "25,50000,NYC,1\n"
    "30,,LA,0\n"
    "35,70000,,1\n"
    "40,80000,NYC,0\n"
)


def test_load_csv_basic():
    p = DatasetProfiler()
    profile = p.load_csv(SAMPLE_CSV, "test.csv")
    assert profile["rows"] == 4
    assert profile["columns"] == 4
    assert "age" in profile["column_names"]


def test_detects_missing_values():
    p = DatasetProfiler()
    profile = p.load_csv(SAMPLE_CSV, "test.csv")
    assert profile["total_missing"] == 2


def test_detects_target_column():
    p = DatasetProfiler()
    profile = p.load_csv(SAMPLE_CSV, "test.csv")
    assert profile["target_guess"] == "target"


def test_numeric_and_categorical_columns():
    p = DatasetProfiler()
    profile = p.load_csv(SAMPLE_CSV, "test.csv")
    assert "age" in profile["numeric_columns"]
    assert "city" in profile["categorical_columns"]


def test_column_info_structure():
    p = DatasetProfiler()
    profile = p.load_csv(SAMPLE_CSV, "test.csv")
    col = next(c for c in profile["column_info"] if c["name"] == "income")
    assert "missing" in col
    assert "missing_pct" in col
    assert "dtype" in col
    assert col["missing"] == 1


def test_context_string():
    p = DatasetProfiler()
    p.load_csv(SAMPLE_CSV, "test.csv")
    ctx = p.get_context_string()
    assert "test.csv" in ctx
    assert "4 rows" in ctx


def test_is_loaded_flag():
    p = DatasetProfiler()
    assert not p.is_loaded
    p.load_csv(SAMPLE_CSV, "test.csv")
    assert p.is_loaded


def test_get_column_list():
    p = DatasetProfiler()
    p.load_csv(SAMPLE_CSV, "test.csv")
    cols = p.get_column_list()
    assert cols == ["age", "income", "city", "target"]


def test_empty_csv():
    p = DatasetProfiler()
    empty = _csv_bytes("a,b,c\n")
    profile = p.load_csv(empty, "empty.csv")
    assert profile["rows"] == 0
    assert profile["columns"] == 3
