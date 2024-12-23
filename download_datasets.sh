# Tạo thư mục data nếu chưa tồn tại
mkdir -p data

# Di chuyển vào thư mục data
cd data

echo "Downloading datasets from Kaggle..."

# Download và giải nén CelebA-Spoof
echo "Downloading CelebA-Spoof dataset..."
kaggle datasets download -d n24q02m/celeba-spoof-face-anti-spoofing-dataset -p CelebA_Spoof_dataset --unzip

# Download và giải nén CATI-FAS 
echo "Downloading CATI-FAS dataset..."
kaggle datasets download -d n24q02m/cati-fas-face-anti-spoofing-dataset -p CATI_FAS_dataset --unzip

# Download và giải nén LCC-FASD
echo "Downloading LCC-FASD dataset..."  
kaggle datasets download -d n24q02m/lcc-fasd-face-anti-spoofing-dataset -p LCC_FASD_dataset --unzip

# Download và giải nén NUAAA
echo "Downloading NUAAA dataset..."
kaggle datasets download -d n24q02m/nuaaa-face-anti-spoofing-dataset -p NUAAA_dataset --unzip

# Download và giải nén Zalo-AIC
echo "Downloading Zalo-AIC dataset..."
kaggle datasets download -d n24q02m/zalo-aic-face-anti-spoofing-dataset -p Zalo_AIC_dataset --unzip

echo "All datasets downloaded and extracted successfully!"

# Hiển thị cấu trúc thư mục
echo "Directory structure:"
tree -L 2