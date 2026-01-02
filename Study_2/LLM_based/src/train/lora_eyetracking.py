"""
LoRA fine-tuning for the eye-tracking + text AI-detection dataset.
"""

import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from datasets import Dataset, DatasetDict, load_from_disk
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG_PATH = PROJECT_ROOT / "configs" / "base.yaml"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "lora_eyetracking.yaml"
DATASET_NAME = "eyetracking_ai_detection"
DATASET_SUBDIR = "dataset_eyetracking"
YES_NO_PATTERN = re.compile(r"\b(yes|no)\b", re.IGNORECASE)


def load_yaml(path: Path) -> Dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def merge_dicts(base: Dict, override: Dict) -> Dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge_dicts(out[k], v)
        else:
            out[k] = v
    return out


def resolve_path(config: Dict, key: str, default: str) -> Path:
    raw_path = config.get(key, default)
    path = Path(raw_path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def extract_yes_no(text: str) -> str:
    match = YES_NO_PATTERN.search(text)
    if not match:
        return "No"
    token = match.group(1).lower()
    return "Yes" if token == "yes" else "No"


def build_train_text(example: Dict, tokenizer) -> str:
    messages = example.get("messages")
    if messages and hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
    user = example.get("user_prompt") or ""
    assistant = (example.get("assistant") or "").strip()
    conv = [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
    return f"{user}\n\n{assistant}"


def build_infer_prompt(example: Dict, tokenizer) -> str:
    messages = example.get("messages")
    prompt = example.get("user_prompt") or ""
    if messages and hasattr(tokenizer, "apply_chat_template"):
        user_only = [m for m in messages if m.get("role") == "user"]
        if not user_only and messages:
            user_only = [messages[0]]
        try:
            prompt = tokenizer.apply_chat_template(
                user_only,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            prompt = tokenizer.apply_chat_template(
                user_only,
                tokenize=False,
                add_generation_prompt=True,
            )
    return prompt


def keep_new_generated(gen_text: str, prompt: str) -> str:
    gen_text = gen_text.strip()
    if prompt and gen_text.startswith(prompt):
        return gen_text[len(prompt) :].strip()
    return gen_text


def choose_batch_size(requested: int, name: str) -> int:
    if requested >= 8:
        print(f"{name}={requested} may be large; ensure GPU memory is sufficient.")
    return max(1, requested)


def tokenize_supervised(ds: Dataset, tokenizer, max_length: int) -> Dataset:
    def _tokenize(example):
        prompt = build_infer_prompt(example, tokenizer)
        answer = (example.get("assistant") or "").strip()
        answer_with_eos = answer + (tokenizer.eos_token or "")

        prompt_ids = tokenizer(
            prompt,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors=None,
        )["input_ids"]

        tokenized = tokenizer(
            prompt + answer_with_eos,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors=None,
        )

        labels = []
        prompt_len = min(len(prompt_ids), len(tokenized["input_ids"]))
        for idx, tok_id in enumerate(tokenized["input_ids"]):
            if tok_id == tokenizer.pad_token_id or idx < prompt_len:
                labels.append(-100)
            else:
                labels.append(tok_id)

        tokenized["labels"] = labels
        return tokenized

    return ds.map(_tokenize, batched=False, remove_columns=ds.column_names)


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


def evaluate_generation(model, tokenizer, ds: Dataset, max_new_tokens: int, temperature: float, top_p: float, repetition_penalty: float) -> Dict:
    preds: List[str] = []
    labels: List[str] = []

    was_training = model.training
    model.eval()
    for ex in ds:
        prompt = build_infer_prompt(ex, tokenizer)
        raw_label = (ex.get("assistant") or "").strip()
        labels.append("Yes" if raw_label.lower() == "yes" else "No")

        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.inference_mode():
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
        gen_clean = keep_new_generated(gen_text, prompt)
        pred_label = extract_yes_no(gen_clean)
        preds.append(pred_label)

        running_accuracy, running_f1 = compute_accuracy_f1(preds, labels)
        print(
            json.dumps(
                {
                    "idx": len(preds) - 1,
                    "pred": pred_label,
                    "label": labels[-1],
                    "generated": gen_clean,
                    "running_accuracy": running_accuracy,
                    "running_f1": running_f1,
                },
                ensure_ascii=False,
            )
        )

    accuracy, f1 = compute_accuracy_f1(preds, labels)
    if was_training:
        model.train()
    return {"accuracy": accuracy, "f1": f1, "samples": len(labels)}


def build_compute_metrics(tokenizer, eval_ds: Dataset, gen_cfg: Dict, model):
    def _compute_metrics(eval_pred):
        _ = eval_pred  
        res = evaluate_generation(
            model=model,
            tokenizer=tokenizer,
            ds=eval_ds,
            max_new_tokens=int(gen_cfg.get("max_new_tokens", 8)),
            temperature=float(gen_cfg.get("temperature", 0.0)),
            top_p=float(gen_cfg.get("top_p", 1.0)),
            repetition_penalty=float(gen_cfg.get("repetition_penalty", 1.0)),
        )
        return {"accuracy": res["accuracy"], "f1": res["f1"]}

    return _compute_metrics


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


class EvalLoggerCallback(TrainerCallback):
    """Log eval metrics and whether a new best checkpoint was saved."""

    def __init__(self):
        self.best_f1_seen = None
        self.best_checkpoint = None

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        current_f1 = metrics.get("eval_f1")
        current_acc = metrics.get("eval_accuracy")

        improved = False
        if current_f1 is not None:
            if self.best_f1_seen is None or current_f1 > self.best_f1_seen:
                improved = True
                self.best_f1_seen = current_f1
                self.best_checkpoint = state.best_model_checkpoint

        best_f1 = self.best_f1_seen if self.best_f1_seen is not None else state.best_metric
        print(
            json.dumps(
                {
                    "event": "eval_end",
                    "step": state.global_step,
                    "eval_accuracy": current_acc,
                    "eval_f1": current_f1,
                    "best_f1": best_f1,
                    "best_checkpoint": state.best_model_checkpoint,
                    "new_best_saved": improved,
                },
                ensure_ascii=False,
            )
        )
        return control


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for eye-tracking AI-detection.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH), help="Path to lora config.")
    parser.add_argument("--base_config", type=str, default=str(BASE_CONFIG_PATH), help="Path to base config.")
    parser.add_argument("--max_train_samples", type=int, default=None, help="Optional cap on train samples.")
    parser.add_argument("--max_eval_samples", type=int, default=None, help="Optional cap on val samples.")
    args = parser.parse_args()

    base_cfg = load_yaml(Path(args.base_config))
    lora_cfg = load_yaml(Path(args.config))
    cfg = merge_dicts(base_cfg, lora_cfg)

    model_name = cfg.get("model_name")
    tokenizer_name = cfg.get("tokenizer_name") or model_name
    trust_remote_code = bool(cfg.get("trust_remote_code", False))

    train_cfg = cfg.get("training", {})
    lora_params = cfg.get("lora", {})
    gen_cfg = cfg.get("generation", {})

    if not model_name:
        raise SystemExit("model_name missing; set it in base.yaml or lora_eyetracking.yaml.")

    output_dir = Path(train_cfg.get("output_dir", PROJECT_ROOT / "outputs" / "lora_eyetracking"))
    output_dir = output_dir if output_dir.is_absolute() else PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(train_cfg.get("seed", 42))

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        device_map="auto",
    )

    lora_config = LoraConfig(
        r=lora_params.get("r", 8),
        lora_alpha=lora_params.get("alpha", 16),
        lora_dropout=lora_params.get("dropout", 0.05),
        bias=lora_params.get("bias", "none"),
        target_modules=lora_params.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    if bool(train_cfg.get("gradient_checkpointing", False)) and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if bool(train_cfg.get("gradient_checkpointing", False)) and hasattr(model, "config"):
        model.config.use_cache = False

    output_data_path = resolve_path(cfg, "output_data_path", "dataset/processed_data")
    split_path = output_data_path / DATASET_SUBDIR / f"{DATASET_NAME}_splits"
    if not split_path.exists():
        raise SystemExit(f"Split dataset not found at {split_path}. Run datasetCreate_eyetracking.py first.")

    print(f"Loading datasets from {split_path}")
    raw_ds: DatasetDict = load_from_disk(split_path)
    train_ds = raw_ds["train"]
    val_ds = raw_ds["val"]
    test_ds = raw_ds["test"]

    if args.max_train_samples:
        train_ds = train_ds.select(range(min(args.max_train_samples, len(train_ds))))
    if args.max_eval_samples:
        val_ds = val_ds.select(range(min(args.max_eval_samples, len(val_ds))))

    max_length = int(train_cfg.get("max_seq_length", 1024))
    print(f"Tokenizing with max length {max_length}")
    train_tok = tokenize_supervised(train_ds, tokenizer, max_length)
    val_tok = tokenize_supervised(val_ds, tokenizer, max_length)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    per_device_train_batch_size = choose_batch_size(
        int(train_cfg.get("per_device_train_batch_size", 1)), "per_device_train_batch_size"
    )
    per_device_eval_batch_size = choose_batch_size(
        int(train_cfg.get("per_device_eval_batch_size", 1)), "per_device_eval_batch_size"
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=float(train_cfg.get("num_train_epochs", 3)),
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=int(train_cfg.get("gradient_accumulation_steps", 1)),
        learning_rate=float(train_cfg.get("learning_rate", 2e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
        warmup_ratio=float(train_cfg.get("warmup_ratio", 0.03)),
        logging_steps=int(train_cfg.get("logging_steps", 10)),
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=int(train_cfg.get("eval_steps", 200)),
        save_steps=int(train_cfg.get("save_steps", 200)),
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        eval_accumulation_steps=int(train_cfg.get("eval_accumulation_steps", 1)),
        fp16=train_cfg.get("fp16", True),
        bf16=train_cfg.get("bf16", False),
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", False),
        report_to=train_cfg.get("report_to", []),
    )

    compute_metrics = build_compute_metrics(tokenizer, val_ds, gen_cfg, model)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=[EvalLoggerCallback()],
    )

    print("Starting training...")
    trainer.train()
    print("Training complete. Best model loaded.")

    val_metrics = trainer.evaluate()
    print("Validation metrics:", json.dumps(val_metrics, indent=2))

    print("Running test evaluation...")
    test_metrics = evaluate_generation(
        model=model,
        tokenizer=tokenizer,
        ds=test_ds,
        max_new_tokens=int(gen_cfg.get("max_new_tokens", 8)),
        temperature=float(gen_cfg.get("temperature", 0.0)),
        top_p=float(gen_cfg.get("top_p", 1.0)),
        repetition_penalty=float(gen_cfg.get("repetition_penalty", 1.0)),
    )
    print("Test metrics:", json.dumps(test_metrics, indent=2))

    best_model_path = Path(trainer.state.best_model_checkpoint or training_args.output_dir)
    print(f"Best model saved at: {best_model_path}")


if __name__ == "__main__":
    main()
