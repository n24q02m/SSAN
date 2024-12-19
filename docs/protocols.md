# Protocols

## Protocol 1: Large-Scale Training

- Training: CelebA-Spoof (90% split)
  - Size: ~65 GB
  - Images: ~478K images (329K live + 622K spoof)
  - Rationale: Dataset lớn nhất, đa dạng nhất, phù hợp cho huấn luyện ban đầu

- Validation: CelebA-Spoof (10% split)
  - Size: ~7.2 GB
  - Images: ~53K images (37K live + 74K spoof)

- Testing: CATI-FAS
  - Size: 40.83 GB
  - Images: 18K images (2K live + 16K spoof)
  - Rationale: Dataset lớn thứ 2, tỷ lệ live:spoof chênh lệch cao (1:8)

## Protocol 2: Multi-Domain Training

- Training: CelebA-Spoof (50%) + NUAAA
  - Size: ~36.5 GB
  - Images: ~249K CelebA + 12.6K NUAAA
  - Rationale: Kết hợp dataset lớn và dataset cân bằng

- Validation: LCC-FASD
  - Size: 936.8 MB
  - Images: 2.9K images (223 live + 2.6K spoof)

- Testing: CATI-FAS
  - Size: 40.83 GB
  - Images: 18K images (2K live + 16K spoof)

## Protocol 3: Balanced Training

- Training: NUAAA + Zalo-AIC (90% split)
  - Size: ~1.4 GB
  - Images: ~16K images
  - Rationale: Các dataset nhỏ hơn nhưng cân bằng về tỷ lệ live:spoof

- Validation: Zalo-AIC (10% split)
  - Size: ~113 MB
  - Images: ~1K images

- Testing: CATI-FAS
  - Size: 40.83 GB  
  - Images: 18K images

## Protocol 4: Cross-Domain Testing

- Training: CelebA-Spoof + NUAAA + LCC-FASD
  - Size: ~73.5 GB
  - Images: ~484K images
  - Rationale: Huấn luyện trên nhiều domain khác nhau

- Validation: CelebA-Spoof validation split
  - Size: ~7.2 GB
  - Images: ~53K images

- Testing: Zalo-AIC
  - Size: 1.13 GB
  - Images: 5.2K images (2.6K live + 2.5K spoof)
  - Rationale: Test khả năng tổng quát hóa trên domain hoàn toàn mới
