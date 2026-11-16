"""End-to-end run on the synthetic file."""

from __future__ import annotations

import json
from pathlib import Path

from turnover.cli import main


def test_full_run_writes_report_and_beats_guessing(raw_path: Path, tmp_path: Path) -> None:
    assert main(["run", "--data", str(raw_path), "--out", str(tmp_path)]) == 0
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["test_years"] == [2014, 2015]
    # The synthetic data makes early tenure risky, so every model must find that.
    assert all(m["roc_auc"] > 0.6 for m in summary["metrics"].values())
    html = (tmp_path / "report.html").read_text()
    assert "Customer Service" in html and "{{" not in html
    for name in ("exits_by_year", "by_tenure", "by_department", "survival", "capture"):
        assert (tmp_path / "figures" / f"{name}.png").stat().st_size > 0


def test_missing_data_returns_error_code(tmp_path: Path) -> None:
    assert main(["run", "--data", str(tmp_path / "missing.csv"), "--out", str(tmp_path)]) == 2
