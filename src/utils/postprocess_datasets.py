import os
import zipfile
import multiprocessing as mp
from pathlib import Path
from tqdm import tqdm
from functools import partial
import cv2

# Dataset paths
DATASETS = {
    "CelebA_Spoof": "./data/CelebA_Spoof_dataset",
    "CATI_FAS": "./data/CATI_FAS_dataset", 
    "LCC_FASD": "./data/LCC_FASD_dataset",
    "NUAAA": "./data/NUAAA_dataset",
    "Zalo_AIC": "./data/Zalo_AIC_dataset"
}

def check_and_remove_file(img_file, min_bbox_ratio=0.05):
    """Check and remove image if no bbox exists or bbox is too small
    Args:
        img_file: Path to image file
        min_bbox_ratio: Minimum ratio of bbox dimension to image dimension
    """
    try:
        img_path = Path(img_file)
        bb_path = img_path.parent / f"{img_path.stem}_BB.txt"
        
        # Check if bbox file exists
        if not bb_path.exists():
            img_path.unlink()
            return 1, "no_bbox"
            
        # Read image dimensions
        img = cv2.imread(str(img_path))
        if img is None:
            img_path.unlink()
            return 1, "invalid_image"
            
        img_h, img_w = img.shape[:2]
        
        # Read bbox dimensions
        with open(bb_path, 'r') as f:
            x, y, w, h, _ = map(float, f.read().strip().split())
        
        # Calculate ratios
        w_ratio = w / img_w
        h_ratio = h / img_h
        
        # Remove if bbox is too small
        if w_ratio < min_bbox_ratio or h_ratio < min_bbox_ratio:
            img_path.unlink()
            bb_path.unlink()
            return 1, "small_bbox"
            
        return 0, "kept"
        
    except Exception as e:
        print(f"Error checking {img_file}: {str(e)}")
        return 0, "error"

def filter_images(dataset_path, min_bbox_ratio=0.05):
    """Filter out invalid images and those with small bboxes"""
    print(f"\nFiltering images in {dataset_path}...")
    stats = {
        'live': {'total': 0, 'no_bbox': 0, 'invalid_image': 0, 'small_bbox': 0, 'error': 0},
        'spoof': {'total': 0, 'no_bbox': 0, 'invalid_image': 0, 'small_bbox': 0, 'error': 0}
    }
    
    # Use all CPU cores
    num_processes = mp.cpu_count()
    print(f"Using {num_processes} processes")
    
    # Process live and spoof folders
    for folder in ['live', 'spoof']:
        folder_path = Path(dataset_path) / folder
        if not folder_path.exists():
            continue
            
        # Get all image files
        image_files = []
        for ext in ['.jpg', '.png']:
            image_files.extend(list(folder_path.glob(f'*{ext}')))
        
        stats[folder]['total'] = len(image_files)
        if not image_files:
            continue
            
        print(f"\nChecking {folder} folder...")
        
        # Process in parallel
        with mp.Pool(num_processes) as pool:
            results = list(tqdm(
                pool.imap(partial(check_and_remove_file, min_bbox_ratio=min_bbox_ratio), 
                         image_files),
                total=len(image_files),
                desc="Filtering files"
            ))
            
        # Update statistics
        for removed, reason in results:
            if removed:
                stats[folder][reason] += 1
    
    # Print statistics
    print("\nFiltering statistics:")
    total_removed = 0
    for folder in ['live', 'spoof']:
        removed = sum(stats[folder][key] for key in ['no_bbox', 'invalid_image', 'small_bbox'])
        kept = stats[folder]['total'] - removed
        total_removed += removed
        
        print(f"\n{folder.upper()} images:")
        print(f"Total: {stats[folder]['total']}")
        print(f"Removed - No bbox: {stats[folder]['no_bbox']}")
        print(f"Removed - Invalid image: {stats[folder]['invalid_image']}")
        print(f"Removed - Small bbox: {stats[folder]['small_bbox']}")
        print(f"Removed - Total: {removed}")
        print(f"Kept: {kept}")
        
    return total_removed

def rename_files_parallel(args):
    """Helper function for parallel file renaming"""
    file, new_name = args
    try:
        file.rename(new_name)
        return True
    except Exception as e:
        print(f"Error renaming {file}: {str(e)}")
        return False

def rename_files(dataset_path):
    """Rename files to 7 digits then back to 6 digits using parallel processing"""
    print(f"\nRenaming files in {dataset_path}...")
    
    # Get live count for spoof start index
    live_path = Path(dataset_path) / 'live'
    live_count = len(list(live_path.glob('*.*'))) // 2 if live_path.exists() else 0
    
    # Use all CPU cores
    num_processes = mp.cpu_count()
    
    for folder in ['live', 'spoof']:
        folder_path = Path(dataset_path) / folder
        if not folder_path.exists():
            continue
            
        # Group files by base name
        file_groups = {}
        for file in folder_path.iterdir():
            base = file.stem.split('_')[0]
            if base not in file_groups:
                file_groups[base] = []
            file_groups[base].append(file)
            
        # First rename to 7 digits
        rename_tasks = []
        for i, (_, group) in enumerate(file_groups.items()):
            new_base = f"{(i+1):07d}"
            for file in group:
                parts = file.stem.split('_')
                extension = file.suffix
                new_name = f"{new_base}{'_BB' if len(parts) > 1 else ''}{extension}"
                new_path = file.parent / new_name
                rename_tasks.append((file, new_path))
                
        print(f"\nConverting {folder} to 7 digits...")
        with mp.Pool(num_processes) as pool:
            list(tqdm(
                pool.imap(rename_files_parallel, rename_tasks),
                total=len(rename_tasks),
                desc="Renaming files"
            ))
            
        # Then rename back to 6 digits
        file_groups = {}
        for file in folder_path.iterdir():
            base = file.stem.split('_')[0]
            if base not in file_groups:
                file_groups[base] = []
            file_groups[base].append(file)
            
        start_idx = 1 if folder == 'live' else live_count + 1
        
        rename_tasks = []
        for i, (_, group) in enumerate(file_groups.items()):
            new_base = f"{(start_idx + i):06d}"
            for file in group:
                parts = file.stem.split('_')
                extension = file.suffix
                new_name = f"{new_base}{'_BB' if len(parts) > 1 else ''}{extension}"
                new_path = file.parent / new_name
                rename_tasks.append((file, new_path))
                
        print(f"Converting {folder} back to 6 digits...")
        with mp.Pool(num_processes) as pool:
            list(tqdm(
                pool.imap(rename_files_parallel, rename_tasks),
                total=len(rename_tasks),
                desc="Renaming files"
            ))

def create_zips(dataset_path):
    """Create zip files for live/spoof folders"""
    print(f"\nCreating zip files for {dataset_path}...")
    
    zip_tasks = []
    for folder in ['live', 'spoof']:
        folder_path = Path(dataset_path) / folder
        if not folder_path.exists():
            continue
            
        output_zip = folder_path.parent / f"{folder}.zip"
        zip_tasks.append((folder_path, output_zip))
        
    # Zip folders in parallel
    with mp.Pool(min(len(zip_tasks), mp.cpu_count())) as pool:
        results = list(tqdm(
            pool.imap(zip_folder, zip_tasks),
            total=len(zip_tasks),
            desc="Creating zips"
        ))
    
    return all(results)

def zip_folder(args):
    """Helper function for parallel zipping"""
    folder_path, output_zip = args
    try:
        with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
            for file in Path(folder_path).iterdir():
                zf.write(file, file.name)
        return True
    except Exception as e:
        print(f"Error zipping {folder_path}: {str(e)}")
        return False

def main():
    for name, path in DATASETS.items():
        if not os.path.exists(path):
            print(f"\nSkipping {name} - directory not found")
            continue
            
        print(f"\nProcessing {name}...")
        
        # Filter images
        removed = filter_images(path, min_bbox_ratio=0.05)
        
        if removed > 0:
            # Rename files if any were removed
            rename_files(path)
            
        # Create zip files
        if create_zips(path):
            print(f"\nDataset {name} is ready for manual upload to Kaggle")
            print(f"Zip files location: {path}")
            print("Please upload live.zip and spoof.zip manually through the Kaggle website")
        else:
            print(f"\nError processing {name}")

if __name__ == '__main__':
    main()