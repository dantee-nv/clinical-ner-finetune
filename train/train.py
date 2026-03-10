"""QLoRA fine-tuning script for clinical NER instruction extraction."""

from __future__ import annotations

import argparse
import inspect
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class TrainConfig:
    """Serializable training configuration."""

    model_name: str = "microsoft/Phi-3-mini-4k-instruct"
    train_file: str = "data/processed/train.jsonl"
    val_file: str = "data/processed/val.jsonl"
    output_dir: str = "outputs/phi3-clinical-ner"
    max_seq_length: int = 768
    num_train_epochs: float = 1.0
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    gradient_checkpointing: bool = True
    logging_steps: int = 10
    eval_steps: int = 50
    save_steps: int = 50
    save_total_limit: int = 2
    seed: int = 42
    max_train_samples: int = 0
    max_eval_samples: int = 0
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load YAML configuration from disk."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise ImportError("pyyaml is required to read training config files.") from exc

    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError("YAML config must be a key-value mapping.")
    return raw


def _coerce_config(raw: dict[str, Any]) -> TrainConfig:
    """Convert dictionary values to TrainConfig."""
    defaults = asdict(TrainConfig())
    defaults.update(raw)
    return TrainConfig(**defaults)


def _import_training_dependencies() -> dict[str, Any]:
    """Import training dependencies lazily to keep module import lightweight."""
    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
        set_seed,
    )
    from trl import SFTTrainer

    return {
        "torch": torch,
        "load_dataset": load_dataset,
        "LoraConfig": LoraConfig,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "BitsAndBytesConfig": BitsAndBytesConfig,
        "TrainingArguments": TrainingArguments,
        "set_seed": set_seed,
        "SFTTrainer": SFTTrainer,
    }


def _resolve_path(path_str: str) -> Path:
    """Resolve relative paths against current working directory."""
    return Path(path_str).expanduser()


def _render_messages(messages: list[dict[str, str]], tokenizer: Any) -> str:
    """Render chat messages to text for SFTTrainer."""
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

    rendered: list[str] = []
    for message in messages:
        role = message.get("role", "unknown")
        content = message.get("content", "")
        rendered.append(f"{role.upper()}: {content}")
    return "\n".join(rendered)


def _prepare_dataset(dataset_split: Any, tokenizer: Any, max_samples: int) -> Any:
    """Map JSONL rows to text field expected by SFTTrainer."""
    if max_samples > 0:
        dataset_split = dataset_split.select(range(min(max_samples, len(dataset_split))))

    def mapper(row: dict[str, Any]) -> dict[str, str]:
        messages = row.get("messages", [])
        return {"text": _render_messages(messages, tokenizer)}

    columns_to_remove = list(dataset_split.column_names)
    return dataset_split.map(mapper, remove_columns=columns_to_remove)


def _infer_lora_targets(model: Any) -> list[str]:
    """Infer practical LoRA target modules from available linear layers."""
    candidate_suffixes = {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "qkv_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "gate_up_proj",
    }

    discovered: set[str] = set()
    for name, module in model.named_modules():
        class_name = module.__class__.__name__.lower()
        if "linear" not in class_name:
            continue
        suffix = name.split(".")[-1]
        if suffix in candidate_suffixes:
            discovered.add(suffix)

    ordered = [suffix for suffix in candidate_suffixes if suffix in discovered]
    return ordered or ["q_proj", "v_proj"]


def train(config: TrainConfig) -> Path:
    """Execute QLoRA fine-tuning and return adapter output path."""
    deps = _import_training_dependencies()
    torch = deps["torch"]

    deps["set_seed"](config.seed)

    train_path = _resolve_path(config.train_file)
    val_path = _resolve_path(config.val_file)
    output_path = _resolve_path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(
            "Training files not found. Run data/curate.py first to create train/val JSONL files."
        )

    bnb_config = deps["BitsAndBytesConfig"](
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    tokenizer = deps["AutoTokenizer"].from_pretrained(
        config.model_name,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = deps["AutoModelForCausalLM"].from_pretrained(
        config.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False

    lora_targets = _infer_lora_targets(model)
    peft_config = deps["LoraConfig"](
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=lora_targets,
    )

    data_files = {
        "train": str(train_path),
        "validation": str(val_path),
    }
    dataset_dict = deps["load_dataset"]("json", data_files=data_files)

    train_dataset = _prepare_dataset(dataset_dict["train"], tokenizer, config.max_train_samples)
    val_dataset = _prepare_dataset(dataset_dict["validation"], tokenizer, config.max_eval_samples)

    training_args_kwargs = {
        "output_dir": str(output_path),
        "num_train_epochs": config.num_train_epochs,
        "per_device_train_batch_size": config.per_device_train_batch_size,
        "per_device_eval_batch_size": config.per_device_eval_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "gradient_checkpointing": config.gradient_checkpointing,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "warmup_ratio": config.warmup_ratio,
        "lr_scheduler_type": config.lr_scheduler_type,
        "fp16": True,
        "bf16": False,
        "logging_steps": config.logging_steps,
        "eval_steps": config.eval_steps,
        "save_steps": config.save_steps,
        "save_total_limit": config.save_total_limit,
        "report_to": "none",
        "seed": config.seed,
        "optim": "paged_adamw_8bit",
    }

    ta_signature = inspect.signature(deps["TrainingArguments"].__init__)
    if "evaluation_strategy" in ta_signature.parameters:
        training_args_kwargs["evaluation_strategy"] = "steps"
    else:
        training_args_kwargs["eval_strategy"] = "steps"

    training_args = deps["TrainingArguments"](**training_args_kwargs)

    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "train_dataset": train_dataset,
        "eval_dataset": val_dataset,
        "args": training_args,
        "peft_config": peft_config,
    }

    sft_signature = inspect.signature(deps["SFTTrainer"].__init__)
    sft_params = sft_signature.parameters

    # TRL API changed across versions; support both old and new constructor names.
    if "tokenizer" in sft_params:
        trainer_kwargs["tokenizer"] = tokenizer
    if "processing_class" in sft_params:
        trainer_kwargs["processing_class"] = tokenizer
    if "dataset_text_field" in sft_params:
        trainer_kwargs["dataset_text_field"] = "text"
    if "max_seq_length" in sft_params:
        trainer_kwargs["max_seq_length"] = config.max_seq_length
    if "packing" in sft_params:
        trainer_kwargs["packing"] = False

    trainer = deps["SFTTrainer"](**trainer_kwargs)

    trainer.train()

    adapter_dir = output_path / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    (output_path / "run_config.json").write_text(
        json.dumps(asdict(config), indent=2),
        encoding="utf-8",
    )

    return adapter_dir


def build_arg_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser for training script."""
    parser = argparse.ArgumentParser(description="Fine-tune Phi-3 Mini with QLoRA on curated clinical NER data.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("train/config.yaml"),
        help="Path to YAML training config.",
    )
    parser.add_argument("--train_file", type=str, default=None, help="Optional train JSONL override.")
    parser.add_argument("--val_file", type=str, default=None, help="Optional val JSONL override.")
    parser.add_argument("--output_dir", type=str, default=None, help="Optional output directory override.")
    return parser


def main() -> int:
    """CLI entrypoint for model fine-tuning."""
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        config_dict = _load_yaml(args.config)
        config = _coerce_config(config_dict)

        if args.train_file:
            config.train_file = args.train_file
        if args.val_file:
            config.val_file = args.val_file
        if args.output_dir:
            config.output_dir = args.output_dir

        adapter_dir = train(config)
    except ImportError as exc:
        print(f"Training failed due to missing dependency: {exc}")
        print("Install dependencies with: pip install -r requirements.txt")
        return 1
    except Exception as exc:
        print(f"Training failed: {exc}")
        return 1

    print(f"Training complete. Adapter saved to: {adapter_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
