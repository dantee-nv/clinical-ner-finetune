"""Dataset curation pipeline for clinical NER instruction tuning.

This module loads a public biomedical/clinical NER dataset, maps labels into a
simple target schema (problems, treatments, tests), filters low-quality
examples, performs deterministic train/val/test splitting with scikit-learn,
and writes JSONL files for supervised instruction fine-tuning.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable

SYSTEM_PROMPT = (
    "You are a clinical NLP model. Extract named medical entities and return valid JSON."
)
TARGET_KEYS = ("problems", "treatments", "tests")


def _import_train_test_split() -> Any:
    """Import scikit-learn split utility lazily."""
    try:
        from sklearn.model_selection import train_test_split
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise ImportError("scikit-learn is required for dataset splitting.") from exc
    return train_test_split


def _import_load_dataset() -> Any:
    """Import Hugging Face datasets loader lazily."""
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise ImportError("The datasets package is required for data curation.") from exc
    return load_dataset


def clean_note(text: str) -> str:
    """Normalize note text by collapsing whitespace and removing null bytes."""
    normalized = text.replace("\x00", " ")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def _dedupe_preserve_order(items: Iterable[str]) -> list[str]:
    """Deduplicate strings while preserving insertion order."""
    seen: set[str] = set()
    deduped: list[str] = []
    for item in items:
        normalized = clean_note(item)
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(normalized)
    return deduped


def _ensure_schema(entities_dict: dict[str, list[str]]) -> dict[str, list[str]]:
    """Guarantee all target keys exist and values are deduplicated string lists."""
    normalized: dict[str, list[str]] = {key: [] for key in TARGET_KEYS}
    for key in TARGET_KEYS:
        values = entities_dict.get(key, [])
        if not isinstance(values, list):
            continue
        normalized[key] = _dedupe_preserve_order([str(value) for value in values])
    return normalized


def format_example(note_text: str, entities_dict: dict[str, list[str]]) -> dict[str, Any]:
    """Format one supervised chat-style training example."""
    entities = _ensure_schema(entities_dict)
    assistant_payload = json.dumps(entities, ensure_ascii=False)

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Clinical note:\n{clean_note(note_text)}"},
            {"role": "assistant", "content": assistant_payload},
        ]
    }


def is_valid_example(note: str, entities: dict[str, list[str]], min_note_chars: int = 20) -> bool:
    """Apply simple quality filters for curated examples."""
    cleaned_note = clean_note(note)
    if len(cleaned_note) < min_note_chars:
        return False

    normalized_entities = _ensure_schema(entities)
    has_any_entities = any(len(values) > 0 for values in normalized_entities.values())
    return has_any_entities


def _join_tokens(tokens: list[str]) -> str:
    """Join tokenized text into a readable sentence-like string."""
    text = " ".join(tokens)
    text = re.sub(r"\s+([,.;:!?%\)])", r"\1", text)
    text = re.sub(r"([\(\[\{])\s+", r"\1", text)
    text = re.sub(r"\s+'", "'", text)
    return clean_note(text)


def _map_label_to_schema(label: str, dataset_name: str) -> str | None:
    """Map source dataset labels to target schema keys."""
    normalized = label.lower().strip()

    if normalized in {"o", "", "none"}:
        return None

    # Dataset-specific shortcuts.
    if dataset_name == "ncbi_disease":
        return "problems"
    if dataset_name == "bc5cdr":
        if "disease" in normalized:
            return "problems"
        if any(token in normalized for token in ("chemical", "drug")):
            return "treatments"

    # Generic mapping fallback.
    if any(token in normalized for token in ("disease", "problem", "condition", "symptom", "disorder")):
        return "problems"
    if any(token in normalized for token in ("chemical", "drug", "medication", "treatment", "therapy")):
        return "treatments"
    if any(token in normalized for token in ("test", "lab", "procedure", "exam", "imaging")):
        return "tests"
    return None


def _decode_tags(tags: list[Any], ner_tag_names: list[str] | None) -> list[str]:
    """Convert integer class ids into label strings when class names are available."""
    decoded: list[str] = []
    for tag in tags:
        if isinstance(tag, int) and ner_tag_names and 0 <= tag < len(ner_tag_names):
            decoded.append(ner_tag_names[tag])
        else:
            decoded.append(str(tag))
    return decoded


def _bio_tags_to_entities(
    tokens: list[str],
    tags: list[str],
    dataset_name: str,
) -> dict[str, list[str]]:
    """Extract entity spans from BIO tags and map into target schema."""
    entities: dict[str, list[str]] = {key: [] for key in TARGET_KEYS}

    current_tokens: list[str] = []
    current_category: str | None = None

    def flush_current() -> None:
        nonlocal current_tokens, current_category
        if current_tokens and current_category:
            entities[current_category].append(_join_tokens(current_tokens))
        current_tokens = []
        current_category = None

    for token, tag in zip(tokens, tags):
        label = tag.strip()
        if label in {"O", ""}:
            flush_current()
            continue

        if "-" in label:
            prefix, raw_label = label.split("-", maxsplit=1)
        else:
            prefix, raw_label = "B", label

        mapped_category = _map_label_to_schema(raw_label, dataset_name)
        if mapped_category is None:
            flush_current()
            continue

        if prefix == "B" or current_category != mapped_category:
            flush_current()
            current_tokens = [token]
            current_category = mapped_category
            continue

        if prefix == "I" and current_category == mapped_category:
            current_tokens.append(token)
            continue

        flush_current()

    flush_current()

    return _ensure_schema(entities)


def _extract_ner_tag_names(dataset_split: Any) -> list[str] | None:
    """Try to extract class label names from datasets.Features metadata."""
    features = getattr(dataset_split, "features", None)
    if not features:
        return None

    candidate_feature = features.get("ner_tags") or features.get("tags")
    if candidate_feature is None:
        return None

    class_label = getattr(candidate_feature, "feature", None)
    names = getattr(class_label, "names", None)
    if names and isinstance(names, list):
        return [str(name) for name in names]

    # Some datasets expose ClassLabel directly.
    names = getattr(candidate_feature, "names", None)
    if names and isinstance(names, list):
        return [str(name) for name in names]

    return None


def _extract_from_entity_dicts(
    row: dict[str, Any],
    dataset_name: str,
) -> dict[str, list[str]]:
    """Extract entities from row['entities'] if present as list of dicts."""
    entities: dict[str, list[str]] = {key: [] for key in TARGET_KEYS}
    raw_entities = row.get("entities")
    if not isinstance(raw_entities, list):
        return entities

    for entry in raw_entities:
        if not isinstance(entry, dict):
            continue
        text = clean_note(str(entry.get("text", "")))
        label = str(entry.get("label") or entry.get("type") or "")
        mapped = _map_label_to_schema(label, dataset_name)
        if text and mapped:
            entities[mapped].append(text)

    return _ensure_schema(entities)


def _extract_from_passages(
    row: dict[str, Any],
    dataset_name: str,
) -> tuple[str, dict[str, list[str]]]:
    """Extract note text and entities from BigBio-style `passages` rows."""
    passages = row.get("passages")
    if not isinstance(passages, list):
        return "", {key: [] for key in TARGET_KEYS}

    note_parts: list[str] = []
    entities: dict[str, list[str]] = {key: [] for key in TARGET_KEYS}

    for passage in passages:
        if not isinstance(passage, dict):
            continue
        passage_text = clean_note(str(passage.get("text", "")))
        if passage_text:
            note_parts.append(passage_text)

        raw_entities = passage.get("entities")
        if not isinstance(raw_entities, list):
            continue

        for entity in raw_entities:
            if not isinstance(entity, dict):
                continue
            label = str(entity.get("type") or entity.get("label") or "")
            mapped = _map_label_to_schema(label, dataset_name)
            if mapped is None:
                continue

            raw_text = entity.get("text")
            if isinstance(raw_text, list):
                entity_texts = [clean_note(str(value)) for value in raw_text if str(value).strip()]
            else:
                entity_texts = [clean_note(str(raw_text))] if str(raw_text).strip() else []

            for text in entity_texts:
                if text:
                    entities[mapped].append(text)

    note_text = clean_note(" ".join(note_parts))
    return note_text, _ensure_schema(entities)


def _row_to_example(
    row: dict[str, Any],
    dataset_name: str,
    ner_tag_names: list[str] | None,
    min_note_chars: int,
) -> dict[str, Any] | None:
    """Convert a raw dataset row into one formatted instruction example."""
    tokens = row.get("tokens")

    if isinstance(tokens, list) and tokens:
        token_texts = [str(token) for token in tokens]
        note_text = _join_tokens(token_texts)

        raw_tags = row.get("ner_tags") or row.get("tags") or row.get("labels")
        if isinstance(raw_tags, list) and len(raw_tags) == len(token_texts):
            decoded_tags = _decode_tags(raw_tags, ner_tag_names)
            entities = _bio_tags_to_entities(token_texts, decoded_tags, dataset_name)
        else:
            entities = {key: [] for key in TARGET_KEYS}
    elif isinstance(row.get("passages"), list):
        note_text, entities = _extract_from_passages(row, dataset_name)
    else:
        text_value = row.get("text") or row.get("document") or row.get("sentence")
        note_text = clean_note(str(text_value or ""))
        entities = _extract_from_entity_dicts(row, dataset_name)

    if not is_valid_example(note_text, entities, min_note_chars=min_note_chars):
        return None

    return format_example(note_text, entities)


def build_dataset(
    rows: Iterable[dict[str, Any]],
    dataset_name: str,
    min_note_chars: int = 20,
    ner_tag_names: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Build formatted instruction examples from raw dataset rows."""
    examples: list[dict[str, Any]] = []
    for row in rows:
        example = _row_to_example(
            row=row,
            dataset_name=dataset_name,
            ner_tag_names=ner_tag_names,
            min_note_chars=min_note_chars,
        )
        if example is not None:
            examples.append(example)
    return examples


def _candidate_dataset_attempts(dataset_name: str) -> list[tuple[str, str | None]]:
    """Return load_dataset attempts for a preferred dataset name."""
    if dataset_name == "bc5cdr":
        return [
            ("bc5cdr", None),
            ("bigbio/bc5cdr", "source"),
            ("bigbio/bc5cdr", "bc5cdr_source"),
            ("bigbio/bc5cdr", "bigbio_kb"),
        ]
    return [(dataset_name, None)]


def _resolve_dataset_split(dataset_obj: Any) -> Any:
    """Select a practical split from Dataset or DatasetDict."""
    if hasattr(dataset_obj, "keys"):
        # DatasetDict-like object.
        for split_name in ("train", "validation", "test"):
            if split_name in dataset_obj:
                return dataset_obj[split_name]
        first_key = next(iter(dataset_obj.keys()))
        return dataset_obj[first_key]
    return dataset_obj


def load_public_dataset(preferences: list[str]) -> tuple[Any, str]:
    """Load the first available public dataset from an ordered preference list."""
    load_dataset = _import_load_dataset()

    errors: list[str] = []
    for preferred_name in preferences:
        dataset_name = preferred_name.strip()
        if not dataset_name:
            continue

        for name, config in _candidate_dataset_attempts(dataset_name):
            try:
                if config:
                    dataset_obj = load_dataset(name, config, trust_remote_code=True)
                else:
                    dataset_obj = load_dataset(name, trust_remote_code=True)
                split = _resolve_dataset_split(dataset_obj)
                return split, dataset_name
            except Exception as exc:  # pragma: no cover - network/environment-specific
                errors.append(f"{name}{'/' + config if config else ''}: {exc}")

    error_summary = "\n".join(errors)
    raise RuntimeError(
        "Unable to load any requested dataset. Attempts:\n"
        f"{error_summary}"
    )


def split_examples(
    examples: list[dict[str, Any]],
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Split examples into train/val/test = 80/10/10 using scikit-learn."""
    if len(examples) < 5:
        raise ValueError("Need at least 5 examples to create train/val/test splits.")

    train_test_split = _import_train_test_split()
    train_examples, temp_examples = train_test_split(
        examples,
        test_size=0.2,
        random_state=seed,
        shuffle=True,
    )
    val_examples, test_examples = train_test_split(
        temp_examples,
        test_size=0.5,
        random_state=seed,
        shuffle=True,
    )
    return train_examples, val_examples, test_examples


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write dictionaries to JSONL with UTF-8 encoding."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    """Create argument parser for curation CLI."""
    parser = argparse.ArgumentParser(description="Curate a public clinical NER dataset into instruction JSONL.")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory where train/val/test JSONL files will be written.",
    )
    parser.add_argument(
        "--dataset_preference",
        type=str,
        default="medical_ner,ncbi_disease,bc5cdr",
        help="Comma-separated dataset preference order.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for deterministic splitting.",
    )
    parser.add_argument(
        "--min_note_chars",
        type=int,
        default=20,
        help="Minimum note length required for an example.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="Optional cap on loaded raw rows (0 means no cap).",
    )
    return parser


def main() -> int:
    """CLI entrypoint for curation pipeline."""
    parser = build_arg_parser()
    args = parser.parse_args()

    preferences = [value.strip() for value in args.dataset_preference.split(",") if value.strip()]
    try:
        raw_split, dataset_name = load_public_dataset(preferences)
    except Exception as exc:
        print(f"Curation failed while loading dataset: {exc}")
        print("Tip: verify internet access and dataset availability, or try a different dataset_preference.")
        return 1

    if args.max_samples > 0:
        max_samples = min(args.max_samples, len(raw_split))
        raw_split = raw_split.select(range(max_samples))

    ner_tag_names = _extract_ner_tag_names(raw_split)
    examples = build_dataset(
        rows=raw_split,
        dataset_name=dataset_name,
        min_note_chars=args.min_note_chars,
        ner_tag_names=ner_tag_names,
    )

    train_examples, val_examples, test_examples = split_examples(examples, seed=args.seed)

    write_jsonl(args.output_dir / "train.jsonl", train_examples)
    write_jsonl(args.output_dir / "val.jsonl", val_examples)
    write_jsonl(args.output_dir / "test.jsonl", test_examples)

    print(f"Loaded dataset: {dataset_name}")
    print(f"Curated examples: {len(examples)}")
    print(f"Train/Val/Test: {len(train_examples)}/{len(val_examples)}/{len(test_examples)}")
    print(f"Wrote files to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
