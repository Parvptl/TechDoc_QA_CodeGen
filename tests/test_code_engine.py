from core.code_engine import CodeEngine


def test_code_engine_rejects_unsafe_code():
    engine = CodeEngine()
    result = engine.execute("import subprocess\nprint('x')")
    assert result["success"] is False


def test_code_engine_executes_simple_code():
    engine = CodeEngine(timeout=5)
    result = engine.execute("x = 2 + 2\nprint(x)")
    assert result["success"] is True
    assert "4" in result["stdout"]
