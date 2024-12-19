# Protocols

## Protocol 1: Large-Scale Training

- Training: CelebA-Spoof (90% split)
  - Size: ~70 GB
  - Images: ~506K images
  - Rationale: Largest and most diverse dataset, good for learning robust features

- Validation: CelebA-Spoof (10% split)
  - Size: ~8 GB
  - Images: ~56K images

- Testing: CATI-FAS
  - Size: 41.69 GB
  - Images: 18.3K images
  - Rationale: Second largest dataset, good for real-world evaluation

## Protocol 2: Multi-Domain Training

- Training: CelebA-Spoof (50%) + LCC-FASD + NUAAA
  - Size: ~44 GB
  - Images: ~300K images
  - Rationale: Mix of large and medium datasets with different characteristics

- Testing: CATI-FAS
  - Size: 41.69 GB
  - Images: 18.3K images

### Protocol 3: Balanced Small-Scale Training

- Training: LCC-FASD + NUAAA + Zalo-AIC
  - Size: ~6.7 GB
  - Images: ~36.5K images
  - Rationale: More balanced training across smaller datasets

- Testing: CATI-FAS
  - Size: 41.69 GB
  -Images: 18.3K images

### Protocol 4: Cross-Domain Testing

- Training: CelebA-Spoof + LCC-FASD + NUAAA
  - Size: ~83.7 GB
  - Images: ~593K images
  - Rationale: Large-scale training with diverse domains

- Testing: Zalo-AIC
  - Size: 1.14 GB
  - Images: 5.2K images
  - Rationale: Test generalization to a completely different domain
