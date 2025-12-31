# LLM_based Project README

## Installation

First, install the required dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

All default experiment configurations are located in the `configs/` directory.

Please update the following field in the config file you use:

```yaml
model_name: "Qwen3-8B"
```

Replace `"Qwen3-8B"` with the path or name of the LLM you want to use.

## Dataset Preparation

Before training or evaluation, you need to create the datasets.

The raw data is located in:

```
dataset/raw_data/
```

Run the following scripts to generate processed datasets for LLM usage:

```bash
python src/datasetCreate/datasetCreate_textdata.py

python src/datasetCreate/datasetCreate_eyetracking.py
```

* `datasetCreate_textdata.py`: creates the text-only dataset

* `datasetCreate_eyetracking.py`: creates the eye-tracking + text dataset

The processed data will be saved to:

```
dataset/processed_data/
```

## Zero-Shot Evaluation

After dataset creation, you can run zero-shot inference:

```bash
python src/evaluate/zeroshot_text.py

python src/evaluate/zeroshot_eyetracking.py
```

* `zeroshot_text.py`: zero-shot evaluation on text-only data

* `zeroshot_eyetracking.py`: zero-shot evaluation on eye-tracking + text data

## LoRA Fine-Tuning and Inference

To perform LoRA fine-tuning and inference, run:

```bash
python src/train/lora_text.py

python src/train/lora_eyetracking.py
```

* `lora_text.py`: LoRA fine-tuning on text-only data

* `lora_eyetracking.py`: LoRA fine-tuning on eye-tracking + text data

## Notes

* Make sure datasets are created before running evaluation or training.

* Configuration files control all major experiment settings.

* Scripts are designed to be run independently for text-only and multimodal experiments.




