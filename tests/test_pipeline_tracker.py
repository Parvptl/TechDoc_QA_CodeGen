from core.pipeline_tracker import PipelineTracker


def test_pipeline_skip_warning():
    tracker = PipelineTracker()
    warnings = tracker.check_prerequisites(6)
    assert warnings


def test_pipeline_progress_marks_stages():
    tracker = PipelineTracker()
    tracker.mark_completed(2)
    tracker.mark_completed(3)
    assert 2 in tracker.completed_stages
    assert 3 in tracker.completed_stages
