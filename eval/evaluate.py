"""Evaluation script for clinical NER extraction on held-out JSONL data."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

TARGET_KEYS = ("problems", "treatments", "tests")
SYSTEM_PROMPT = (
    "You are a clinical NLP model. Extract named medical entities and return valid JSON."
)


class Predictor(Protocol):
    """Protocol for prediction backends used by evaluation."""

    def predict(self, note: str) -> dict[str, list[str]]:
        """Predict normalized entity schema for one note."""


@dataclass
class Example:
    """Ground-truth evaluation example parsed from test JSONL."""

    note: str
    entities: dict[str, list[str]]


def _import_metric_dependencies() -> tuple[Any, Any]:
    """Import scikit-learn utilities lazily."""
    try:
        from sklearn.metrics import precision_recall_fscore_support
        from sklearn.preprocessing import MultiLabelBinarizer
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise ImportError("scikit-learn is required for evaluation metrics.") from exc
    return precision_recall_fscore_support, MultiLabelBinarizer


def _import_model_dependencies() -> dict[str, Any]:
    """Import heavy model dependencies lazily."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    return {
        "torch": torch,
        "PeftModel": PeftModel,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "BitsAndBytesConfig": BitsAndBytesConfig,
    }


def normalize_entity(value: str) -> str:
    """Normalize entity strings for exact-match scoring."""
    return " ".join(value.lower().split()).strip()


def normalize_entity_list(values: list[str]) -> list[str]:
    """Deduplicate and normalize entity lists while preserving order."""
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        cleaned = normalize_entity(value)
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        output.append(cleaned)
    return output


def _default_schema() -> dict[str, list[str]]:
    """Return empty target extraction schema."""
    return {key: [] for key in TARGET_KEYS}


def _extract_note(user_content: str) -> str:
    """Extract note text from user message content."""
    prefix = "Clinical note:\n"
    return user_content[len(prefix) :].strip() if user_content.startswith(prefix) else user_content.strip()


def _extract_json_substring(text: str) -> str | None:
    """Extract first balanced JSON object from arbitrary model text."""
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def parse_entity_payload(payload: Any) -> dict[str, list[str]]:
    """Normalize payload dictionary into target schema."""
    normalized = _default_schema()
    if not isinstance(payload, dict):
        return normalized

    for key in TARGET_KEYS:
        values = payload.get(key, [])
        if not isinstance(values, list):
            continue
        strings = [str(value) for value in values if isinstance(value, str)]
        normalized[key] = normalize_entity_list(strings)

    return normalized


def parse_model_response(text: str) -> dict[str, list[str]]:
    """Parse model output text into target schema if valid JSON is present."""
    maybe_json = _extract_json_substring(text)
    if maybe_json is None:
        return _default_schema()

    try:
        payload = json.loads(maybe_json)
    except json.JSONDecodeError:
        return _default_schema()

    return parse_entity_payload(payload)


def load_test_examples(path: Path, max_examples: int = 0) -> list[Example]:
    """Load held-out test examples from JSONL messages format."""
    examples: list[Example] = []

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue

            row = json.loads(stripped)
            messages = row.get("messages", [])
            if not isinstance(messages, list):
                continue

            role_map = {
                message.get("role"): message.get("content")
                for message in messages
                if isinstance(message, dict)
            }
            if "user" not in role_map or "assistant" not in role_map:
                continue

            note = _extract_note(str(role_map["user"]))
            try:
                gold_payload = json.loads(str(role_map["assistant"]))
            except json.JSONDecodeError:
                continue

            entities = parse_entity_payload(gold_payload)
            examples.append(Example(note=note, entities=entities))

            if max_examples > 0 and len(examples) >= max_examples:
                break

    return examples


def compute_label_metrics(
    y_true: list[list[str]],
    y_pred: list[list[str]],
) -> dict[str, Any]:
    """Compute micro-averaged precision/recall/F1 for one label group."""
    precision_recall_fscore_support, MultiLabelBinarizer = _import_metric_dependencies()

    mlb = MultiLabelBinarizer()
    fit_data = [set(values) for values in y_true + y_pred]
    if not fit_data:
        fit_data = [set()]

    mlb.fit(fit_data)
    true_bin = mlb.transform([set(values) for values in y_true])
    pred_bin = mlb.transform([set(values) for values in y_pred])

    precision, recall, f1, _ = precision_recall_fscore_support(
        true_bin,
        pred_bin,
        average="micro",
        zero_division=0,
    )

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "support": int(true_bin.sum()),
        "num_classes": int(len(mlb.classes_)),
    }


def score_predictions(
    gold_entities: list[dict[str, list[str]]],
    pred_entities: list[dict[str, list[str]]],
    focus_labels: tuple[str, ...] = ("problems", "treatments"),
) -> dict[str, Any]:
    """Compute per-label and overall entity extraction metrics."""
    if len(gold_entities) != len(pred_entities):
        raise ValueError("Gold and prediction lengths must match.")

    by_label: dict[str, dict[str, Any]] = {}
    for label in focus_labels:
        true_lists = [normalize_entity_list(item.get(label, [])) for item in gold_entities]
        pred_lists = [normalize_entity_list(item.get(label, [])) for item in pred_entities]
        by_label[label] = compute_label_metrics(true_lists, pred_lists)

    y_true_all: list[list[str]] = []
    y_pred_all: list[list[str]] = []
    for gold, pred in zip(gold_entities, pred_entities):
        gold_set: list[str] = []
        pred_set: list[str] = []
        for label in focus_labels:
            gold_set.extend(f"{label}:{value}" for value in normalize_entity_list(gold.get(label, [])))
            pred_set.extend(f"{label}:{value}" for value in normalize_entity_list(pred.get(label, [])))
        y_true_all.append(gold_set)
        y_pred_all.append(pred_set)

    overall = compute_label_metrics(y_true_all, y_pred_all)

    exact_match = 0
    for gold, pred in zip(gold_entities, pred_entities):
        if all(
            set(normalize_entity_list(gold.get(label, [])))
            == set(normalize_entity_list(pred.get(label, [])))
            for label in focus_labels
        ):
            exact_match += 1

    return {
        "by_label": by_label,
        "overall": overall,
        "exact_match_rate": float(exact_match / len(gold_entities)) if gold_entities else 0.0,
        "num_examples": len(gold_entities),
        "focus_labels": list(focus_labels),
    }


class AdapterPredictor:
    """Inference predictor using base model + LoRA adapter."""

    def __init__(self, base_model: str, adapter_dir: Path, max_new_tokens: int = 192) -> None:
        deps = _import_model_dependencies()
        torch = deps["torch"]

        quant_config = deps["BitsAndBytesConfig"](
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

        self.tokenizer = deps["AutoTokenizer"].from_pretrained(base_model, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base = deps["AutoModelForCausalLM"].from_pretrained(
            base_model,
            quantization_config=quant_config,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model = deps["PeftModel"].from_pretrained(base, str(adapter_dir))
        # Disable KV cache for compatibility across transformers/remote Phi-3 code versions.
        self.model.config.use_cache = False
        self.model.eval()

        self.torch = torch
        self.device = next(self.model.parameters()).device
        self.max_new_tokens = max_new_tokens

    def _build_prompt(self, note: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Clinical note:\n{note}"},
        ]

        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        return (
            f"SYSTEM: {SYSTEM_PROMPT}\n"
            f"USER: Clinical note:\n{note}\n"
            "ASSISTANT:"
        )

    def predict(self, note: str) -> dict[str, list[str]]:
        prompt = self._build_prompt(note)
        tokenized = self.tokenizer(prompt, return_tensors="pt")
        tokenized = {key: value.to(self.device) for key, value in tokenized.items()}

        with self.torch.no_grad():
            generated = self.model.generate(
                **tokenized,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=0.0,
                use_cache=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        prompt_len = tokenized["input_ids"].shape[-1]
        generated_tokens = generated[0][prompt_len:]
        raw = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return parse_model_response(raw)


class EmptyPredictor:
    """Placeholder predictor useful for quick pipeline smoke checks."""

    def predict(self, note: str) -> dict[str, list[str]]:  # noqa: ARG002
        return _default_schema()


def build_predictor(
    predictor_name: str,
    base_model: str,
    adapter_dir: Path | None,
    max_new_tokens: int,
) -> Predictor:
    """Build predictor backend based on CLI selection."""
    if predictor_name == "adapter":
        if adapter_dir is None:
            raise ValueError("--adapter_dir is required for adapter predictor.")
        return AdapterPredictor(base_model=base_model, adapter_dir=adapter_dir, max_new_tokens=max_new_tokens)

    if predictor_name == "empty":
        return EmptyPredictor()

    raise ValueError(f"Unsupported predictor: {predictor_name}")


def evaluate(
    test_file: Path,
    predictor: Predictor,
    focus_labels: tuple[str, ...],
    max_examples: int,
) -> dict[str, Any]:
    """Run predictions over held-out test set and compute evaluation metrics."""
    examples = load_test_examples(test_file, max_examples=max_examples)
    if not examples:
        raise ValueError("No valid examples found in test file.")

    gold = [example.entities for example in examples]
    preds = [predictor.predict(example.note) for example in examples]

    metrics = score_predictions(gold, preds, focus_labels=focus_labels)
    metrics["test_file"] = str(test_file)
    metrics["evaluated_at_utc"] = datetime.now(timezone.utc).isoformat()
    return metrics


def build_arg_parser() -> argparse.ArgumentParser:
    """Create CLI parser for evaluation script."""
    parser = argparse.ArgumentParser(description="Evaluate clinical NER extraction on held-out JSONL data.")
    parser.add_argument("--test_file", type=Path, default=Path("data/processed/test.jsonl"))
    parser.add_argument(
        "--base_model",
        type=str,
        default="microsoft/Phi-3-mini-4k-instruct",
        help="Base model ID used during fine-tuning.",
    )
    parser.add_argument(
        "--adapter_dir",
        type=Path,
        default=None,
        help="Adapter directory for model-based evaluation.",
    )
    parser.add_argument(
        "--predictor",
        type=str,
        choices=["adapter", "empty"],
        default="adapter",
        help="Prediction backend. Use 'empty' for smoke checks without a model.",
    )
    parser.add_argument(
        "--focus_labels",
        type=str,
        default="problems,treatments",
        help="Comma-separated labels to score.",
    )
    parser.add_argument("--max_examples", type=int, default=0, help="Optional evaluation row cap.")
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=192,
        help="Generation budget for adapter predictor.",
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        default=Path("eval/results.json"),
        help="Where to save JSON metrics.",
    )
    return parser


def main() -> int:
    """CLI entrypoint for held-out evaluation."""
    parser = build_arg_parser()
    args = parser.parse_args()

    focus_labels = tuple(label.strip() for label in args.focus_labels.split(",") if label.strip())
    try:
        predictor = build_predictor(
            predictor_name=args.predictor,
            base_model=args.base_model,
            adapter_dir=args.adapter_dir,
            max_new_tokens=args.max_new_tokens,
        )

        results = evaluate(
            test_file=args.test_file,
            predictor=predictor,
            focus_labels=focus_labels,
            max_examples=args.max_examples,
        )
        results["predictor"] = args.predictor
        results["base_model"] = args.base_model
        if args.adapter_dir is not None:
            results["adapter_dir"] = str(args.adapter_dir)

        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        args.output_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    except ImportError as exc:
        print(f"Evaluation failed due to missing dependency: {exc}")
        print("Install dependencies with: pip install -r requirements.txt")
        return 1
    except Exception as exc:
        print(f"Evaluation failed: {exc}")
        return 1

    print("Evaluation complete")
    print(f"Examples: {results['num_examples']}")
    print(f"Overall precision: {results['overall']['precision']:.4f}")
    print(f"Overall recall: {results['overall']['recall']:.4f}")
    print(f"Overall F1: {results['overall']['f1']:.4f}")
    print(f"Saved results to: {args.output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
