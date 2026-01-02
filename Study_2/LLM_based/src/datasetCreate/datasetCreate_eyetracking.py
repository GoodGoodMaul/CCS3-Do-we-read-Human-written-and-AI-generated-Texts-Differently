"""
Build a Qwen3-friendly Yes/No dataset that detects whether a passage was AI-generated using both text and eye-tracking signals.
"""

import ast
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

try:
    from datasets import Dataset, DatasetDict
except ModuleNotFoundError as exc:
    raise SystemExit(
        "datasets dependency is missing. Install with `pip install -U datasets fsspec`."
    ) from exc

try:
    import yaml
except ModuleNotFoundError as exc:
    raise SystemExit("PyYAML dependency is missing. Install with `pip install -U pyyaml`.") from exc

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "base.yaml"
DATASET_NAME = "eyetracking_ai_detection"
DATASET_SUBDIR = "dataset_eyetracking"

FEATURE_NAMES = [
    "Maximum_duration_of_whole_fixations",
    "Average_duration_of_Visit",
    "Average_duration_of_Glances",
    "First-pass_first_fixation_duration",
]
FEATURE_LABELS = ["MaxFix", "AvgVisit", "AvgGlance", "FP1"]

PROMPT_TEMPLATE = (
    "You are an AI-generation detector using BOTH the passage text and human eye-tracking signals.\n"
    "Decide whether the passage was produced by an AI system.\n\n"
    "Answer format:\n"
    "- Output exactly one word: Yes or No\n"
    "- Yes = AI-generated, No = human-written\n\n"
    "Eye-tracking features (milliseconds; NA means missing):\n"
    "- MaxFix: maximum duration of whole fixations on the word\n"
    "- AvgVisit: average duration of visit\n"
    "- AvgGlance: average duration of glances\n"
    "- FP1: first-pass first fixation duration\n\n"
    "Passage:\n{text}\n\n"
    "Word sequence used for alignment (do not re-tokenize):\n{word_sequence}\n\n"
    "Per-word eye-tracking aligned to the word sequence:\n"
    "word | MaxFix, AvgVisit, AvgGlance, FP1\n"
    "{eye_tracking}\n"
)


def load_base_config(config_path: Path) -> Dict:
    with config_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_path(config: Dict, key: str, default: str) -> Path:
    raw_path = config.get(key, default)
    path = Path(raw_path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_literal(value: str):
    if not value:
        return []
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return []


def coerce_float_or_none(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_word_sequence(raw_value: str) -> List[str]:
    words = parse_literal(raw_value)
    return [str(w) for w in words] if isinstance(words, (list, tuple)) else []


def parse_eye_feature_matrix(raw_value: str) -> List[List[float]]:
    matrix = parse_literal(raw_value)
    if not isinstance(matrix, (list, tuple)):
        return []

    parsed: List[List[float]] = []
    for row in matrix:
        if not isinstance(row, (list, tuple)):
            parsed.append([None] * len(FEATURE_NAMES))
            continue
        values = [coerce_float_or_none(val) for val in row[: len(FEATURE_NAMES)]]
        if len(values) < len(FEATURE_NAMES):
            values.extend([None] * (len(FEATURE_NAMES) - len(values)))
        parsed.append(values)
    return parsed


def build_word_eye_features(words: List[str], matrix: List[List[float]]) -> List[Dict]:
    features: List[Dict] = []
    for idx, word in enumerate(words):
        row = matrix[idx] if idx < len(matrix) else [None] * len(FEATURE_NAMES)
        if len(row) < len(FEATURE_NAMES):
            row = list(row) + [None] * (len(FEATURE_NAMES) - len(row))
        row = row[: len(FEATURE_NAMES)]
        entry = {"word": word}
        entry.update({name: row[i] for i, name in enumerate(FEATURE_NAMES)})
        features.append(entry)
    return features


def format_number(value) -> str:
    if value is None:
        return "NA"
    num = coerce_float_or_none(value)
    if num is None:
        return "NA"
    return str(int(num)) if num.is_integer() else f"{num:.2f}"


def format_word_sequence(words: List[str]) -> str:
    return json.dumps(words, ensure_ascii=False)


def format_eye_tracking(features: List[Dict]) -> str:
    lines = []
    for item in features:
        values = ", ".join(
            format_number(item.get(name, None)) for name in FEATURE_NAMES
        )
        lines.append(f"{item.get('word', '')} | {values}")
    return "\n".join(lines)


def build_records(csv_path: Path) -> List[Dict]:
    records: List[Dict] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = (row.get("Text") or "").strip()
            source = (row.get("Source") or "").strip()
            split = (row.get("Split") or "").strip()
            participant = (row.get("Participant") or "").strip()
            toi = (row.get("TOI") or "").strip()
            part_id = (row.get("Part_ID") or "").strip()
            toi_part_key = (row.get("TOI_Part_Key") or "").strip()

            words = parse_word_sequence(row.get("Word_Sequence") or "")
            eye_matrix = parse_eye_feature_matrix(row.get("Eye_Feature_Matrix") or "")
            word_eye_features = build_word_eye_features(words, eye_matrix)

            user_prompt = PROMPT_TEMPLATE.format(
                text=text,
                word_sequence=format_word_sequence(words),
                eye_tracking=format_eye_tracking(word_eye_features),
            )
            label = "Yes" if source.lower() == "ai" else "No"

            records.append(
                {
                    "split": split,
                    "source": source,
                    "participant": participant,
                    "toi": toi,
                    "part_id": part_id,
                    "toi_part_key": toi_part_key,
                    "text": text,
                    "word_sequence": words,
                    "eye_feature_matrix": eye_matrix,
                    "word_eye_features": word_eye_features,
                    "user_prompt": user_prompt,
                    "assistant": label,
                    "messages": [
                        {"role": "user", "content": user_prompt},
                        {"role": "assistant", "content": label},
                    ],
                }
            )
    return records


def build_split_datasets(records: List[Dict]) -> DatasetDict:
    splits = sorted({record["split"] for record in records})
    split_map = {
        split: Dataset.from_list([record for record in records if record["split"] == split])
        for split in splits
    }
    return DatasetDict(split_map)


def save_first_sample(dataset: Dataset, output_path: Path) -> None:
    if len(dataset) == 0:
        raise ValueError("Dataset is empty; cannot save the first sample.")
    output_path.write_text(json.dumps(dataset[0], ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    config = load_base_config(CONFIG_PATH)
    input_dir = resolve_path(config, "input_data_path", "dataset/raw_data")
    output_dir = resolve_path(config, "output_data_path", "dataset/processed_data")

    raw_data_path = input_dir / "data.csv"
    dataset_eye_dir = output_dir / DATASET_SUBDIR
    dataset_eye_dir.mkdir(parents=True, exist_ok=True)

    records = build_records(raw_data_path)
    dataset = Dataset.from_list(records)
    split_datasets = build_split_datasets(records)

    splits_out_path = dataset_eye_dir / f"{DATASET_NAME}_splits"
    first_sample_path = dataset_eye_dir / f"{DATASET_NAME}_first_sample.json"

    split_datasets.save_to_disk(splits_out_path)
    save_first_sample(dataset, first_sample_path)

    split_counts = Counter(record["split"] for record in records)
    print(f"Built dataset with {len(dataset)} samples from {raw_data_path}.")
    print("Split sizes:", dict(split_counts))
    for split_name, split_ds in split_datasets.items():
        print(f"{split_name} subset: {len(split_ds)} samples")
    print(f"Saved split datasets to {splits_out_path}")
    print(f"First sample saved to {first_sample_path}")


if __name__ == "__main__":
    main()
