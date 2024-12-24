# Virtural Machine Setup

**1. Download Git & Clone Repository**

- Download Git from [here](https://github.com/git-for-windows/git/releases/download/v2.47.1.windows.1/Git-2.47.1-64-bit.exe)

- Open Git Bash and set up the username and email
```bash
git config --global user.name "Your Name"
git config --global user.email "Your Email"
```

- Clone the repository
```bash
cd Desktop
git clone https://github.com/n24q02m/SSAN.git
cd SSAN
```

**2. Create a new Conda Environment & Install Dependencies**

- Create a new conda environment
```bash
conda create -n ssan python=3.10
conda activate ssan
```

- Check the GPU, CUDA version for PyTorch installation
```bash
nvcc --version
nvidia-smi
```

- Install dependencies (for ThueGPU's VM)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

**3. Setup Kaggle & Download Datasets**

- Copy folder C:\Users\{username}\.kaggle in local machine to C:\Users\{username}\.kaggle in VM

- Run download_datasets.py to download the datasets
```bash
python download_datasets.py
```

**4. Run main.py**

- Test the main.py file with train mode and small arguments
```bash
python -m src.main --mode train --protocol protocol_3 --epochs 3 --auto_hp --fraction 0.01
```

- Test the main.py file with test mode and small arguments
```bash
python -m src.main --mode test --protocol protocol_3 --fraction 0.01 --checkpoint output\train_{YYYYMMDD_HHMMSS}\checkpoints\best.pth
```

- Run the main.py file with train mode
```bash
python -m src.main --mode train --protocol protocol_1 --epochs 100 --auto_hp --fraction 1.0
```

- Run the main.py file with test mode
```bash
python -m src.main --mode test --protocol protocol_1 --fraction 1.0 --checkpoint output\train_{YYYYMMDD_HHMMSS}\checkpoints\best.pth
```

**5. Upload the Results**

- Create new version for model in Kaggle
```bash
kaggle datasets init -p output\train_{YYYYMMDD_HHMMSS}\checkpoints
```

- Update metadata file in output\train_{YYYYMMDD_HHMMSS}\checkpoints\dataset-metadata.json
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

- Update new version for model to Kaggle
```bash
kaggle datasets version -p output\train_{YYYYMMDD_HHMMSS}\checkpoints -m "Update Model"
```

- Push the results to Github with PAT [Personal Access Token](https://github.com/settings/personal-access-tokens)
```bash
git add .
git commit -m "Update Model"
git push
```
