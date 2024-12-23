import os
import subprocess
from pathlib import Path
from typing import List, Dict

def run_command(command: str) -> None:
    """Execute shell command and print output"""
    print(f"Running: {command}")
    process = subprocess.Popen(
        command.split(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    stdout, stderr = process.communicate()
    
    if stdout:
        print(stdout.decode())
    if stderr:
        print(stderr.decode())
    
    if process.returncode != 0:
        raise RuntimeError(f"Command failed with return code {process.returncode}")

def download_dataset(dataset_name: str, output_dir: str) -> None:
    """Download single dataset from Kaggle"""
    print(f"\nDownloading {dataset_name} dataset...")
    kaggle_command = (
        f"kaggle datasets download -d n24q02m/{dataset_name}-face-anti-spoofing-dataset "
        f"-p {output_dir} --unzip"
    )
    run_command(kaggle_command)

def main():
    # Create data directory if not exists
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Change to data directory
    os.chdir(data_dir)
    print("Downloading datasets from Kaggle...")

    # List of datasets to download
    datasets = {
        "celeba-spoof": "CelebA_Spoof_dataset",
        "cati-fas": "CATI_FAS_dataset",
        "lcc-fasd": "LCC_FASD_dataset", 
        "nuaaa": "NUAAA_dataset",
        "zalo-aic": "Zalo_AIC_dataset"
    }

    # Download each dataset
    for dataset_name, output_dir in datasets.items():
        try:
            download_dataset(dataset_name, output_dir)
        except Exception as e:
            print(f"Error downloading {dataset_name}: {str(e)}")
            continue

    print("\nAll datasets downloaded and extracted successfully!")

    # Show directory structure 
    print("\nDirectory structure:")
    run_command("tree -L 2")

if __name__ == "__main__":
    main()