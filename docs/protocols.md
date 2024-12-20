# Face Anti-Spoofing Testing Protocols

## Protocol 1: Single Large-Scale Dataset

Testing generalization on a single large dataset

- **Training (60%)**: ~319k samples
  - 220k live + 418k spoof
  - ~43.3 GB
- **Validation (20%)**: ~106k samples
  - 73k live + 139k spoof
  - ~14.4 GB
- **Testing (20%)**: ~106k samples
  - 73k live + 139k spoof
  - ~14.4 GB

## Protocol 2: Multi-Scale Training

Testing model performance on mixed large and medium-scale datasets

- **Training**:
  - CelebA-Spoof (30%): ~160k samples (~21.7 GB)
  - CATI-FAS (60%): ~21.6k samples (~24.5 GB)
- **Validation**:
  - CelebA-Spoof (10%): ~53k samples (~7.2 GB)
  - CATI-FAS (20%): ~7.2k samples (~8.2 GB)
- **Testing**:
  - CelebA-Spoof (10%): ~53k samples (~7.2 GB)
  - CATI-FAS (20%): ~7.2k samples (~8.2 GB)

## Protocol 3: Cross-Dataset Evaluation

Testing generalization across multiple medium-scale datasets

- **Training**:
  - CATI-FAS (80%): ~28.8k samples (~32.7 GB)
  - Zalo-AIC (60%): ~6.2k samples (~678 MB)
- **Validation**:
  - CATI-FAS (20%): ~7.2k samples (~8.2 GB)
  - Zalo-AIC (40%): ~4.1k samples (~452 MB)
- **Testing**:
  - LCC-FASD (30%): ~1.7k samples (~281 MB)
  - NUAAA (30%): ~7.6k samples (~117 MB)

## Protocol 4: Domain Generalization

Testing generalization from medium to large-scale domains

- **Training (All Medium Datasets)**:
  - CATI-FAS: ~36k samples (~40.83 GB)
  - LCC-FASD: ~5.7k samples (~937 MB)
  - NUAAA: ~25.2k samples (~391 MB)
  - Zalo-AIC: ~10.3k samples (~1.13 GB)
  - Total: ~77k samples
- **Validation**:
  - CelebA-Spoof (2.5%): ~26.5k samples (~3.6 GB)
- **Testing**:
  - CelebA-Spoof (2.5%): ~26.5k samples (~3.6 GB)
