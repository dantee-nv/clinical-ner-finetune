"""Inference utility for clinical entity extraction with a LoRA adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

SYSTEM_PROMPT = (
    "You are a clinical NLP model. Extract named medical entities and return valid JSON."
)
TARGET_KEYS = ("problems", "treatments", "tests")


def _import_inference_dependencies() -> dict[str, Any]:
    """Import heavy inference dependencies lazily."""
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


def _default_output() -> dict[str, list[str]]:
    """Return an empty target schema."""
    return {key: [] for key in TARGET_KEYS}


def _extract_json_substring(text: str) -> str | None:
    """Extract the first balanced JSON object substring from text."""
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


def _parse_model_json(text: str) -> dict[str, list[str]]:
    """Parse model output into standardized entity schema."""
    extracted = _extract_json_substring(text)
    if extracted is None:
        return _default_output()

    try:
        payload = json.loads(extracted)
    except json.JSONDecodeError:
        return _default_output()

    normalized: dict[str, list[str]] = _default_output()
    if not isinstance(payload, dict):
        return normalized

    for key in TARGET_KEYS:
        values = payload.get(key, [])
        if not isinstance(values, list):
            continue
        cleaned_values: list[str] = []
        seen: set[str] = set()
        for value in values:
            if not isinstance(value, str):
                continue
            cleaned = " ".join(value.split()).strip()
            if not cleaned:
                continue
            lower = cleaned.lower()
            if lower in seen:
                continue
            seen.add(lower)
            cleaned_values.append(cleaned)
        normalized[key] = cleaned_values

    return normalized


def _build_prompt(note_text: str, tokenizer: Any) -> str:
    """Create model prompt using chat template when available."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Clinical note:\n{note_text}"},
    ]

    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    return (
        f"SYSTEM: {SYSTEM_PROMPT}\n"
        f"USER: Clinical note:\n{note_text}\n"
        "ASSISTANT:"
    )


def _load_notes(cli_notes: list[str], notes_file: Path | None) -> list[str]:
    """Load notes from repeated --note flags and optional text file."""
    notes: list[str] = []
    for note in cli_notes:
        cleaned = " ".join(note.split()).strip()
        if cleaned:
            notes.append(cleaned)

    if notes_file:
        with notes_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                cleaned = " ".join(line.split()).strip()
                if cleaned:
                    notes.append(cleaned)

    return notes


def predict_notes(
    base_model: str,
    adapter_dir: Path,
    notes: list[str],
    max_new_tokens: int,
    temperature: float,
) -> list[dict[str, Any]]:
    """Run model inference and return parsed predictions per note."""
    deps = _import_inference_dependencies()
    torch = deps["torch"]

    quant_config = deps["BitsAndBytesConfig"](
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    tokenizer = deps["AutoTokenizer"].from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = deps["AutoModelForCausalLM"].from_pretrained(
        base_model,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = deps["PeftModel"].from_pretrained(base, str(adapter_dir))
    # Disable KV cache for compatibility across transformers/remote Phi-3 code versions.
    model.config.use_cache = False
    model.eval()

    device = next(model.parameters()).device

    results: list[dict[str, Any]] = []
    for note in notes:
        prompt = _build_prompt(note, tokenizer)
        tokenized = tokenizer(prompt, return_tensors="pt")
        tokenized = {key: value.to(device) for key, value in tokenized.items()}

        with torch.no_grad():
            output_ids = model.generate(
                **tokenized,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature,
                use_cache=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        prompt_len = tokenized["input_ids"].shape[-1]
        generated_ids = output_ids[0][prompt_len:]
        raw_response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        parsed = _parse_model_json(raw_response)

        results.append(
            {
                "note": note,
                "raw_response": raw_response.strip(),
                "prediction": parsed,
            }
        )

    return results


def build_arg_parser() -> argparse.ArgumentParser:
    """Create CLI parser for prediction script."""
    parser = argparse.ArgumentParser(description="Run inference with a fine-tuned clinical NER adapter.")
    parser.add_argument(
        "--base_model",
        type=str,
        default="microsoft/Phi-3-mini-4k-instruct",
        help="Base model ID used during fine-tuning.",
    )
    parser.add_argument(
        "--adapter_dir",
        type=Path,
        required=True,
        help="Path to saved LoRA adapter directory.",
    )
    parser.add_argument(
        "--note",
        action="append",
        default=[],
        help="Clinical note text. Repeat this flag for multiple notes.",
    )
    parser.add_argument(
        "--notes_file",
        type=Path,
        default=None,
        help="Optional text file with one clinical note per line.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=192, help="Generation token budget.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    return parser


def main() -> int:
    """CLI entrypoint for local inference."""
    parser = build_arg_parser()
    args = parser.parse_args()

    notes = _load_notes(args.note, args.notes_file)
    if not notes:
        print("No notes provided. Use --note or --notes_file.")
        return 1

    try:
        outputs = predict_notes(
            base_model=args.base_model,
            adapter_dir=args.adapter_dir,
            notes=notes,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
    except ImportError as exc:
        print(f"Inference failed due to missing dependency: {exc}")
        print("Install dependencies with: pip install -r requirements.txt")
        return 1
    except Exception as exc:
        print(f"Inference failed: {exc}")
        return 1

    for index, item in enumerate(outputs, start=1):
        print(f"\n=== Example {index} ===")
        print("Input note:")
        print(item["note"])
        print("\nPredicted entities:")
        print(json.dumps(item["prediction"], indent=2, ensure_ascii=False))
        print("\nRaw model response:")
        print(item["raw_response"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
