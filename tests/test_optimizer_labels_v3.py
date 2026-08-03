from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app.py"
METRICS = ROOT / "src" / "metrics.py"

GROSS_OPTION = "Highest cumulative gross income over common years"
GROSS_AXIS = "Cumulative gross income over common years ($/acre)"
OPERATING_OPTION = (
    "Cumulative highest total return above operating cost over common years"
)
OPERATING_AXIS = (
    "Cumulative total return above operating cost over common years ($/acre)"
)


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_python_files_parse() -> None:
    ast.parse(_text(APP), filename=str(APP))
    ast.parse(_text(METRICS), filename=str(METRICS))


def test_requested_optimizer_labels_are_present() -> None:
    app_text = _text(APP)
    metrics_text = _text(METRICS)

    assert GROSS_OPTION in app_text
    assert GROSS_OPTION in metrics_text
    assert GROSS_AXIS in app_text

    assert OPERATING_OPTION in app_text
    assert OPERATING_OPTION in metrics_text
    assert OPERATING_AXIS in app_text


def test_old_optimizer_labels_are_absent() -> None:
    combined = _text(APP) + "\n" + _text(METRICS)

    assert '"Highest total gross income"' not in combined
    assert '"Highest total return above operating cost"' not in combined
    assert (
        '"Highest cumulative return above operating cost over common years"'
        not in combined
    )
    assert (
        '"Cumulative return above operating cost over common years ($/acre)"'
        not in combined
    )
