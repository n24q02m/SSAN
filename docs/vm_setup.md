# Virtural Machine (Windows x Windows) Setup

**1. Connect to the VM using Remote Desktop Connection**

- Open Remote Desktop Connection, tab Local Resources and select C:\ drive

- Connect to the VM

- Open Anaconda Prompt

- Create temp folder in C:\ drive
```bash
pushd {C:\ drive path}
```

**2. Create a new Conda Environment & Install Dependencies**

- reate a new conda environment
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

**3. Run main.py**

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
