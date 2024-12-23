# Virtural Machine Setup

1. **Check GPU, CUDA & Find the right PyTorch version**

- Check the GPU, CUDA version
```bash
nvcc --version
nvidia-smi
```
- Find the right PyTorch version in [PyTorch](https://pytorch.org/)

1. **Install Github Desktop & Sign In with token**

- [Download Github Desktop](https://desktop.github.com/)
- [Create a personal access token](https://github.com/settings/personal-access-tokens)

3. **Clone the Repository & Install Dependencies**

- Clone the repository using Github Desktop
- **Open CMD in the repository folder** & Install dependencies without conda
```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126
pip install -r requirements.txt
```

4. **Setup Kaggle & Download Datasets**

- Copy C:\Users\{username}\.kaggle\kaggle.json from local machine to VM
- Download the required datasets
```bash
chmod +x download_datasets.sh
./download_datasets.sh
```

5. **Run main.py**

- Test the main.py file with train mode and small arguments
```bash
python -m src.main --mode train --protocol protocol_3 --epochs 3 --auto_hp --hp_trials 3 --fraction 0.01 --no_workers
```

- Test the main.py file with test mode and small arguments
```bash
python -m src.main --mode test --protocol protocol_3 --checkpoints output\train_{YYYYMMDD_HHMMSS}\checkpoints\best.pth --fraction 0.01 --no_workers
```

- Run the main.py file with train mode
```bash
python -m src.main --mode train --protocol protocol_1 --epochs 100 --auto_hp --hp_trials 100 --fraction 1.0 --no_workers
```

- Run the main.py file with test mode
```bash
python -m src.main --mode test --protocol protocol_1 --checkpoints output\train_{YYYYMMDD_HHMMSS}\checkpoints\best.pth --fraction 1.0 --no_workers
```

6. **Upload the trained model to Kaggle**

- Initialize a json file for the model
```bash
kaggle datasets init -p output\train_{YYYYMMDD_HHMMSS}\checkpoints
```
- Edit the dataset-metadata.json file
```json
{
  "title": "SSAN - Face Anti-Spoofing Model",
  "id": "n24q02m/ssan-face-anti-spoofing-model",
  "licenses": [
    {
      "name": "CC0-1.0"
    }
  ]
}
```
- Upload new model version to Kaggle
```bash
kaggle datasets version -p output\train_{YYYYMMDD_HHMMSS}\checkpoints -m "New model version"
```

7. **Git push the changes**

- Open Github Desktop & commit the changes
- Push the changes to the repository
