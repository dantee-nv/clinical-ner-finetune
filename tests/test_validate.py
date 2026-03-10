"""Unit tests for processed JSONL validation."""

from __future__ import annotations

import json
from pathlib import Path

from data.validate import validate_files, validate_row


def _valid_row() -> dict:
    return {
        "messages": [
            {
                "role": "system",
                "content": "You are a clinical NLP model. Extract named medical entities and return valid JSON.",
            },
            {"role": "user", "content": "Clinical note:\nPatient has asthma and uses albuterol."},
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "problems": ["asthma"],
                        "treatments": ["albuterol"],
                        "tests": [],
                    }
                ),
            },
        ]
    }


def test_validate_row_accepts_valid_schema() -> None:
    errors, entities, note_length = validate_row(_valid_row())

    assert errors == []
    assert entities is not None
    assert entities["problems"] == ["asthma"]
    assert note_length > 10


def test_validate_row_flags_invalid_assistant_json() -> None:
    row = _valid_row()
    row["messages"][2]["content"] = "not-json"

    errors, entities, _ = validate_row(row)

    assert errors == ["assistant_invalid_json"]
    assert entities is None


def test_validate_files_reports_summary(tmp_path: Path) -> None:
    valid = _valid_row()
    invalid = _valid_row()
    invalid["messages"][2]["content"] = "{}"

    file_path = tmp_path / "dataset.jsonl"
    with file_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(valid) + "\n")
        handle.write(json.dumps(invalid) + "\n")

    summary = validate_files([file_path])

    assert summary.files_checked == 1
    assert summary.total_rows == 2
    assert summary.valid_rows == 1
    assert summary.invalid_rows == 1
    assert summary.entity_distribution["problems"] == 1
