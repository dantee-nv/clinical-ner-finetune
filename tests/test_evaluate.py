"""Unit tests for evaluation helpers."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from data.validate import validate_files
from eval.evaluate import (
    compute_label_metrics,
    load_test_examples,
    normalize_entity_list,
    score_predictions,
)

SKLEARN_AVAILABLE = importlib.util.find_spec("sklearn") is not None


@pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn is required")
def test_compute_label_metrics_perfect_match() -> None:
    y_true = [["diabetes", "hypertension"], ["asthma"]]
    y_pred = [["diabetes", "hypertension"], ["asthma"]]

    metrics = compute_label_metrics(y_true, y_pred)
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(1.0)


def test_normalize_entity_list_deduplicates_and_normalizes() -> None:
    values = ["  Chest  X-ray", "chest x-ray", "HbA1c", ""]
    assert normalize_entity_list(values) == ["chest x-ray", "hba1c"]


@pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn is required")
def test_compute_label_metrics_handles_zero_class_case() -> None:
    metrics = compute_label_metrics([[], []], [[], []])
    assert metrics["precision"] == pytest.approx(0.0)
    assert metrics["recall"] == pytest.approx(0.0)
    assert metrics["f1"] == pytest.approx(0.0)
    assert metrics["support"] == 0
    assert metrics["num_classes"] == 0


@pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn is required")
def test_integration_smoke_validation_and_evaluation_helpers(tmp_path: Path) -> None:
    row = {
        "messages": [
            {
                "role": "system",
                "content": "You are a clinical NLP model. Extract named medical entities and return valid JSON.",
            },
            {"role": "user", "content": "Clinical note:\nPatient has pneumonia and received azithromycin."},
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "problems": ["pneumonia"],
                        "treatments": ["azithromycin"],
                        "tests": [],
                    }
                ),
            },
        ]
    }

    test_file = tmp_path / "test.jsonl"
    test_file.write_text(json.dumps(row) + "\n", encoding="utf-8")

    validation_summary = validate_files([test_file])
    assert validation_summary.invalid_rows == 0

    examples = load_test_examples(test_file)
    assert len(examples) == 1

    gold = [examples[0].entities]
    pred = [
        {
            "problems": ["pneumonia"],
            "treatments": ["azithromycin"],
            "tests": [],
        }
    ]
    results = score_predictions(gold, pred, focus_labels=("problems", "treatments"))
    assert results["overall"]["f1"] == pytest.approx(1.0)
    assert results["exact_match_rate"] == pytest.approx(1.0)
