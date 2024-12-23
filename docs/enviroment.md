# Enviroment

- CUDA 12.6
- Python 3.10

```shell
conda create -n ssan python=3.10
conda activate ssan
```

- PyTorch

```shell
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126
```

- Other dependencies

```shell
pip install -r requirements.txt
```
