"""Validation script for processed clinical NER instruction datasets."""

from __future__ import annotations

import argparse
import glob
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TARGET_KEYS = ("problems", "treatments", "tests")
REQUIRED_ROLES = ("system", "user", "assistant")


@dataclass
class ValidationSummary:
    """Aggregated validation statistics across one or more JSONL files."""

    files_checked: int
    total_rows: int
    valid_rows: int
    invalid_rows: int
    malformed_json_rows: int
    empty_outputs: int
    average_note_length: float
    entity_distribution: dict[str, int]


def extract_note_from_user_message(content: str) -> str:
    """Extract note text from user prompt content."""
    prefix = "Clinical note:\n"
    if content.startswith(prefix):
        return content[len(prefix) :].strip()
    return content.strip()


def _empty_entity_schema() -> dict[str, list[str]]:
    """Return a blank entity schema dictionary."""
    return {key: [] for key in TARGET_KEYS}


def _normalize_entity_schema(candidate: Any) -> tuple[dict[str, list[str]], list[str]]:
    """Validate and normalize assistant JSON payload."""
    errors: list[str] = []
    schema = _empty_entity_schema()

    if not isinstance(candidate, dict):
        return schema, ["assistant_payload_not_dict"]

    for key in TARGET_KEYS:
        if key not in candidate:
            errors.append(f"missing_key:{key}")
            continue
        if not isinstance(candidate[key], list):
            errors.append(f"key_not_list:{key}")
            continue

        normalized_values: list[str] = []
        for value in candidate[key]:
            if not isinstance(value, str):
                errors.append(f"non_string_entity:{key}")
                continue
            cleaned = " ".join(value.split()).strip()
            if cleaned:
                normalized_values.append(cleaned)

        schema[key] = normalized_values

    return schema, errors


def validate_row(row: Any) -> tuple[list[str], dict[str, list[str]] | None, int]:
    """Validate one JSONL row and return errors + parsed entities + note length."""
    errors: list[str] = []

    if not isinstance(row, dict):
        return ["row_not_object"], None, 0

    messages = row.get("messages")
    if not isinstance(messages, list):
        return ["missing_messages_list"], None, 0

    role_to_content: dict[str, str] = {}
    for message in messages:
        if not isinstance(message, dict):
            errors.append("message_not_object")
            continue

        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            errors.append("invalid_message_fields")
            continue

        role_to_content[role] = content

    for required_role in REQUIRED_ROLES:
        if required_role not in role_to_content:
            errors.append(f"missing_role:{required_role}")

    if errors:
        return errors, None, 0

    note_text = extract_note_from_user_message(role_to_content["user"])
    note_length = len(note_text)

    assistant_raw = role_to_content["assistant"]
    try:
        assistant_payload = json.loads(assistant_raw)
    except json.JSONDecodeError:
        return ["assistant_invalid_json"], None, note_length

    normalized_schema, schema_errors = _normalize_entity_schema(assistant_payload)
    if schema_errors:
        return schema_errors, None, note_length

    return [], normalized_schema, note_length


def validate_files(file_paths: list[Path]) -> ValidationSummary:
    """Validate a set of JSONL files and return summary statistics."""
    total_rows = 0
    valid_rows = 0
    invalid_rows = 0
    malformed_json_rows = 0
    empty_outputs = 0
    note_lengths: list[int] = []
    entity_distribution = {key: 0 for key in TARGET_KEYS}

    for file_path in file_paths:
        with file_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue

                total_rows += 1
                try:
                    row = json.loads(stripped)
                except json.JSONDecodeError:
                    malformed_json_rows += 1
                    invalid_rows += 1
                    continue

                errors, entities, note_length = validate_row(row)
                if errors or entities is None:
                    invalid_rows += 1
                    continue

                valid_rows += 1
                note_lengths.append(note_length)

                row_entity_count = 0
                for key in TARGET_KEYS:
                    count = len(entities[key])
                    entity_distribution[key] += count
                    row_entity_count += count
                if row_entity_count == 0:
                    empty_outputs += 1

    average_note_length = sum(note_lengths) / len(note_lengths) if note_lengths else 0.0

    return ValidationSummary(
        files_checked=len(file_paths),
        total_rows=total_rows,
        valid_rows=valid_rows,
        invalid_rows=invalid_rows,
        malformed_json_rows=malformed_json_rows,
        empty_outputs=empty_outputs,
        average_note_length=average_note_length,
        entity_distribution=entity_distribution,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Create CLI parser for processed dataset validation."""
    parser = argparse.ArgumentParser(description="Validate processed clinical NER JSONL files.")
    parser.add_argument(
        "--input_glob",
        type=str,
        default="data/processed/*.jsonl",
        help="Glob pattern for JSONL files to validate.",
    )
    parser.add_argument(
        "--max_invalid_rows",
        type=int,
        default=0,
        help="Maximum allowed invalid rows before exiting with a non-zero code.",
    )
    return parser


def main() -> int:
    """CLI entrypoint for dataset quality validation."""
    parser = build_arg_parser()
    args = parser.parse_args()

    matched_paths = sorted(Path(path) for path in glob.glob(args.input_glob))
    if not matched_paths:
        raise FileNotFoundError(f"No files matched glob: {args.input_glob}")

    summary = validate_files(matched_paths)

    print("Validation Summary")
    print("------------------")
    print(f"Files checked: {summary.files_checked}")
    print(f"Total rows: {summary.total_rows}")
    print(f"Valid rows: {summary.valid_rows}")
    print(f"Invalid rows: {summary.invalid_rows}")
    print(f"Malformed JSON rows: {summary.malformed_json_rows}")
    print(f"Empty entity outputs: {summary.empty_outputs}")
    print(f"Average note length: {summary.average_note_length:.2f} chars")
    print(
        "Entity distribution: "
        + ", ".join(f"{key}={value}" for key, value in summary.entity_distribution.items())
    )

    if summary.invalid_rows > args.max_invalid_rows:
        print(
            f"Validation failed: invalid rows ({summary.invalid_rows}) "
            f"exceed threshold ({args.max_invalid_rows})."
        )
        return 1

    print("Validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
