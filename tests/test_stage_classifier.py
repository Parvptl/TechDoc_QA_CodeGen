from core.stage_classifier import StageClassifier


def test_classifies_data_loading_query():
    clf = StageClassifier()
    stage, confidence = clf.classify("How do I load a csv into pandas?")
    assert stage == 2
    assert 0.0 <= confidence <= 1.0


def test_stage_name_lookup():
    clf = StageClassifier()
    assert clf.get_stage_name(7) == "Evaluation"
