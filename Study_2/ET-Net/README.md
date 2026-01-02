# ET-Net

This repository contains the code for the ET-Net.

## 📂 File Structure

* **`processed_dataset_final.csv`**: The preprocessed and normalized dataset.
* **`network.py`**: Contains two functions.
* **`main_text_eye.py`**: The main execution file.
* **`run_all.sh`**: Shell script to run the experiments.

## ⚙️ Environment

conda create -n ETNet python=3.10 -y

conda activate ETNet

conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

pip install hydra-core pandas torchmetrics easydict simplejson fvcore wandb

pip install scikit-learn

conda install python-dateutil

pip install transformers==4.57.3 huggingface-hub==0.34.0

pip install ninja einops
