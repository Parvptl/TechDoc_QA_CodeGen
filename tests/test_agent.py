from core.agent import MentorAgent


def test_agent_process_sync_shape():
    agent = MentorAgent()
    result = agent.process_sync("How do I load a CSV file with pandas?")
    assert "stage_num" in result
    assert "stage_name" in result
    assert "confidence" in result


def test_agent_detects_antipattern():
    code = "from sklearn.preprocessing import StandardScaler\nscaler = StandardScaler()\nX = scaler.fit_transform(X)"
    agent = MentorAgent()
    result = agent.process_sync("Check this code", provided_code=code)
    assert isinstance(result["antipattern_warnings"], list)
