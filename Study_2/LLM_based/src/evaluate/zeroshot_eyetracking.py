"""
Zero-shot evaluation on the eye-tracking + text AI-detection dataset.
"""

import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple
import torch
from datasets import DatasetDict, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
import yaml




PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "base.yaml"
DATASET_NAME = "eyetracking_ai_detection"
DATASET_SUBDIR = "dataset_eyetracking"

YES_NO_PATTERN = re.compile(r"\b(yes|no)\b", re.IGNORECASE)


def load_base_config(config_path: Path) -> Dict:
    with config_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_path(config: Dict, key: str, default: str) -> Path:
    raw_path = config.get(key, default)
    path = Path(raw_path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_model_and_tokenizer(model_name: str, tokenizer_name: str, trust_remote_code: bool):
    tok_name = tokenizer_name or model_name
    tokenizer = AutoTokenizer.from_pretrained(tok_name, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=trust_remote_code, device_map="auto"
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer


def extract_yes_no(text: str) -> str:
    match = YES_NO_PATTERN.search(text)
    if not match:
        return "No"
    token = match.group(1).lower()
    return "Yes" if token == "yes" else "No"


def compute_accuracy_f1(preds: List[str], labels: List[str]) -> Tuple[float, float]:
    assert len(preds) == len(labels)
    yes_label = "Yes"
    tp = sum(p == yes_label and l == yes_label for p, l in zip(preds, labels))
    tn = sum(p != yes_label and l != yes_label for p, l in zip(preds, labels))
    fp = sum(p == yes_label and l != yes_label for p, l in zip(preds, labels))
    fn = sum(p != yes_label and l == yes_label for p, l in zip(preds, labels))

    total = len(preds)
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return accuracy, f1


def generate_prediction(
    model,
    tokenizer,
    input_text: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
):
    inputs = tokenizer(input_text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    gen_tokens = outputs[0][inputs["input_ids"].shape[1] :]
    gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
    return gen_text.strip()


def keep_new_generated(gen_text: str, prompt: str) -> str:
    gen_text = gen_text.strip()
    if prompt and gen_text.startswith(prompt):
        return gen_text[len(prompt) :].strip()
    return gen_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Zero-shot evaluation on eye-tracking AI-detection test split.")
    parser.add_argument(
        "--split_path",
        type=str,
        default=None,
        help="Path to saved split datasets (defaults to config output path).",
    )
    parser.add_argument("--max_samples", type=int, default=None, help="Optional cap on test samples.")
    args = parser.parse_args()

    config = load_base_config(CONFIG_PATH)
    eval_cfg = config.get("evaluation", {})
    max_new_tokens = eval_cfg.get("max_new_tokens", 8)
    temperature = eval_cfg.get("temperature", 0.0)
    top_p = eval_cfg.get("top_p", 1.0)
    repetition_penalty = eval_cfg.get("repetition_penalty", 1.0)
    seed = eval_cfg.get("seed", 42)

    random.seed(seed)
    torch.manual_seed(seed)

    model_name = config.get("model_name")
    tokenizer_name = config.get("tokenizer_name")
    trust_remote_code = bool(config.get("trust_remote_code", False))

    if not model_name:
        raise SystemExit("model_name missing from configs/base.yaml.")

    output_dir = resolve_path(config, "output_data_path", "dataset/processed_data")
    default_split_path = output_dir / DATASET_SUBDIR / f"{DATASET_NAME}_splits"
    split_path = Path(args.split_path) if args.split_path else default_split_path

    if not split_path.exists():
        raise SystemExit(
            f"Split dataset not found at {split_path}. Run datasetCreate_eyetracking.py first."
        )

    print(f"Loading split datasets from: {split_path}")
    datasets: DatasetDict = load_from_disk(split_path)
    if "test" not in datasets:
        raise SystemExit("Test split not found in loaded DatasetDict.")
    test_ds = datasets["test"]

    if args.max_samples:
        test_ds = test_ds.select(range(min(args.max_samples, len(test_ds))))

    print(f"Loading model: {model_name}")
    model, tokenizer = load_model_and_tokenizer(model_name, tokenizer_name, trust_remote_code)

    preds: List[str] = []
    labels: List[str] = []

    for example in test_ds:
        prompt = example.get("user_prompt") or ""
        messages = example.get("messages")
        if messages and hasattr(tokenizer, "apply_chat_template"):
            user_only_msgs = [m for m in messages if m.get("role") == "user"]
            if not user_only_msgs and messages:
                user_only_msgs = [messages[0]]
            try:
                prompt = tokenizer.apply_chat_template(
                    user_only_msgs,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                prompt = tokenizer.apply_chat_template(
                    user_only_msgs,
                    tokenize=False,
                    add_generation_prompt=True,
                )
        raw_label = (example.get("assistant") or "").strip()
        labels.append("Yes" if raw_label.lower() == "yes" else "No")

        generated = generate_prediction(
            model=model,
            tokenizer=tokenizer,
            input_text=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
        generated_new = keep_new_generated(generated, prompt)
        pred_label = extract_yes_no(generated_new)
        preds.append(pred_label)

        running_accuracy, running_f1 = compute_accuracy_f1(preds, labels)
        print(
            json.dumps(
                {
                    "idx": len(preds) - 1,
                    "pred": pred_label,
                    "label": labels[-1],
                    "generated": generated_new,
                    "running_accuracy": running_accuracy,
                    "running_f1": running_f1,
                },
                ensure_ascii=False,
            )
        )

    accuracy, f1 = compute_accuracy_f1(preds, labels)
    print(json.dumps({"accuracy": accuracy, "f1": f1, "samples": len(labels)}, indent=2))


if __name__ == "__main__":
    main()
