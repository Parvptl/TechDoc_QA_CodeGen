"""Session learning report export utilities."""
import json
import os
from datetime import datetime
from typing import Dict, Any


class ReportExporter:
    """Builds and saves learning reports in markdown or json."""

    def __init__(self, out_dir: str = "storage/reports"):
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

    def export(self, session_id: str, report_data: Dict[str, Any], fmt: str = "markdown") -> Dict[str, str]:
        fmt_norm = (fmt or "markdown").strip().lower()
        if fmt_norm not in {"markdown", "json"}:
            fmt_norm = "markdown"

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        ext = "md" if fmt_norm == "markdown" else "json"
        filename = f"{session_id}_{timestamp}.{ext}"
        path = os.path.join(self.out_dir, filename)

        if fmt_norm == "json":
            content = json.dumps(report_data, ensure_ascii=True, indent=2)
        else:
            content = self._to_markdown(report_data)

        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

        return {"path": path, "format": fmt_norm, "content": content}

    @staticmethod
    def _to_markdown(report_data: Dict[str, Any]) -> str:
        summary = report_data.get("summary", {})
        trends = report_data.get("trends", {})
        stage_mastery = summary.get("stage_mastery", {})
        stage_dist = summary.get("stage_distribution", {})
        cp_pass = summary.get("checkpoint_passed", 0)
        cp_total = summary.get("checkpoint_total", 0)
        cp_rate = (cp_pass / cp_total * 100.0) if cp_total else 0.0

        lines = [
            "# DS Mentor Pro - Session Learning Report",
            "",
            f"- Session ID: `{summary.get('session_id', 'default')}`",
            f"- Questions asked: {summary.get('questions_asked', 0)}",
            f"- Avg confidence: {summary.get('avg_confidence', 0)}%",
            f"- Avg response latency: {summary.get('avg_response_ms', 0)} ms",
            f"- Quiz attempts: {summary.get('quiz_count', 0)}",
            f"- Avg quiz score: {int(summary.get('avg_quiz_score', 0) * 100)}%",
            f"- Checkpoints passed: {cp_pass}/{cp_total} ({cp_rate:.1f}%)",
            "",
            "## Stage Mastery",
            "",
        ]

        for stage in range(1, 8):
            mastery = float(stage_mastery.get(stage, stage_mastery.get(str(stage), 0.0)))
            bar = "#" * int(round(mastery * 10))
            lines.append(f"- Stage {stage}: `{bar:<10}` {int(mastery * 100)}%")

        lines.extend(["", "## Stage Query Distribution", ""])
        if stage_dist:
            for stage_num in sorted(stage_dist.keys(), key=lambda x: int(x)):
                lines.append(f"- Stage {stage_num}: {stage_dist[stage_num]} queries")
        else:
            lines.append("- No stage activity recorded.")

        lines.extend(["", "## Trend Snapshot", ""])
        lines.append(f"- Confidence points: {len(trends.get('confidence_series', []))}")
        lines.append(f"- Quiz score points: {len(trends.get('quiz_series', []))}")
        lines.append(f"- Checkpoint events: {len(trends.get('checkpoint_series', []))}")

        return "\n".join(lines) + "\n"
