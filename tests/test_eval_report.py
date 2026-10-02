from __future__ import annotations

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "run_eval", Path(__file__).parents[1] / "evals" / "run_eval.py"
)
run_eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_eval)


def test_cost_usd_uses_per_model_prices() -> None:
    cost = run_eval.cost_usd(
        {"claude-opus-5-5": (1_000_000, 100_000), "claude-sonnet-5-5": (500_000, 0)}
    )
    assert cost == 4.0 + 2.0 + 1.0


def test_dataset_is_well_formed() -> None:
    import json

    lines = (Path(__file__).parents[1] / "evals" / "dataset.jsonl").read_text().splitlines()
    items = [json.loads(line) for line in lines if line]
    assert len({i["id"] for i in items}) == len(items) >= 10
    for item in items:
        assert item["finding"] in {"positive", "null"}
        assert item["key_facts"]
        assert any(s.startswith("PMID:") for s in item["expected_sources"])
        assert all(s.startswith(("PMID:", "NCT")) for s in item["expected_sources"])


def test_markdown_report_renders() -> None:
    report = {
        "created_at": "2026-10-01T12:00:00+00:00",
        "config": {"llm_model": "m", "judge_model": "j", "embedding_model": "e"},
        "n": 1,
        "summary": {"fact_recall": 1.0},
        "rows": [
            {
                "id": "select",
                "source_recall": 1.0,
                "cited_expected": True,
                "fact_recall": 1.0,
                "finding_correct": True,
                "grounded": True,
                "cost_usd": 0.12,
            }
        ],
    }
    md = run_eval.to_markdown(report)
    assert "| select | 1.00 | yes | 1.00 | ok | yes | $0.120 |" in md
