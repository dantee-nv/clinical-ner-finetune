"""Unit tests for dataset curation helpers."""

from __future__ import annotations

import json

import pytest

from data.curate import (
    build_dataset,
    clean_note,
    format_example,
    is_valid_example,
    split_examples,
)


def test_clean_note_collapses_whitespace() -> None:
    raw = "  Patient\n\t has   diabetes   "
    assert clean_note(raw) == "Patient has diabetes"


def test_format_example_returns_expected_messages_schema() -> None:
    example = format_example(
        "Patient has diabetes.",
        {"problems": ["diabetes"], "treatments": [], "tests": ["A1c"]},
    )

    assert "messages" in example
    assert [message["role"] for message in example["messages"]] == ["system", "user", "assistant"]

    payload = json.loads(example["messages"][2]["content"])
    assert payload["problems"] == ["diabetes"]
    assert payload["tests"] == ["A1c"]


def test_is_valid_example_filters_short_or_empty() -> None:
    assert not is_valid_example("short", {"problems": ["asthma"], "treatments": [], "tests": []})
    assert not is_valid_example(
        "Patient note with enough characters for threshold.",
        {"problems": [], "treatments": [], "tests": []},
    )
    assert is_valid_example(
        "Patient note with enough characters for threshold.",
        {"problems": ["asthma"], "treatments": [], "tests": []},
    )


def test_build_dataset_from_token_rows() -> None:
    rows = [
        {
            "tokens": ["Patient", "has", "asthma", "and", "uses", "albuterol"],
            "ner_tags": ["O", "O", "B-Disease", "O", "O", "B-Chemical"],
        }
    ]

    examples = build_dataset(rows, dataset_name="bc5cdr", min_note_chars=10)

    assert len(examples) == 1
    payload = json.loads(examples[0]["messages"][2]["content"])
    assert payload["problems"] == ["asthma"]
    assert payload["treatments"] == ["albuterol"]


def test_build_dataset_from_bigbio_passages() -> None:
    rows = [
        {
            "passages": [
                {
                    "text": "Patient with pneumonia treated with azithromycin.",
                    "entities": [
                        {"type": "Disease", "text": ["pneumonia"]},
                        {"type": "Chemical", "text": ["azithromycin"]},
                    ],
                }
            ]
        }
    ]

    examples = build_dataset(rows, dataset_name="bc5cdr", min_note_chars=10)

    assert len(examples) == 1
    payload = json.loads(examples[0]["messages"][2]["content"])
    assert payload["problems"] == ["pneumonia"]
    assert payload["treatments"] == ["azithromycin"]


def test_split_examples_returns_deterministic_80_10_10() -> None:
    pytest.importorskip("sklearn")

    examples = [
        format_example(
            f"Clinical note {idx} with enough detail to keep for training.",
            {"problems": [f"problem-{idx}"], "treatments": [], "tests": []},
        )
        for idx in range(20)
    ]

    train_a, val_a, test_a = split_examples(examples, seed=123)
    train_b, val_b, test_b = split_examples(examples, seed=123)

    assert len(train_a) == 16
    assert len(val_a) == 2
    assert len(test_a) == 2

    assert train_a == train_b
    assert val_a == val_b
    assert test_a == test_b
