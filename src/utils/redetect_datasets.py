import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from insightface.app import FaceAnalysis

# Initialize InsightFace FaceAnalysis with RetinaFace detector
app = FaceAnalysis(
    allowed_modules=["detection"],
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
app.prepare(ctx_id=0, det_size=(640, 640))

# Dataset paths
DATASETS = {
    "CelebA_Spoof": "./data/CelebA_Spoof_dataset",
    "CATI_FAS": "./data/CATI_FAS_dataset",
    "LCC_FASD": "./data/LCC_FASD_dataset",
    "NUAAA": "./data/NUAAA_dataset",
    "Zalo_AIC": "./data/Zalo_AIC_dataset",
}


def remove_old_bbox_files(dataset_path):
    """Remove all existing BB.txt files"""
    print(f"\nRemoving old bbox files from {dataset_path}...")
    bbox_files = list(Path(dataset_path).rglob("*_BB.txt"))
    for f in tqdm(bbox_files):
        f.unlink()


def process_image(img_path):
    """Process single image with RetinaFace and save bbox"""
    try:
        # Read image
        img = cv2.imread(str(img_path))
        if img is None:
            return False

        # Get original dimensions
        real_h, real_w = img.shape[:2]

        # Resize for detection (maintain aspect ratio)
        scale = 640 / max(real_h, real_w)
        if scale < 1:
            new_h, new_w = int(real_h * scale), int(real_w * scale)
            img_resized = cv2.resize(img, (new_w, new_h))
        else:
            img_resized = img
            scale = 1

        # Detect faces on resized image
        faces = app.get(img_resized)

        if len(faces) == 0:
            print(f"No face detected in {img_path}")
            return False

        # Get largest face by bbox area
        areas = [(f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]) for f in faces]
        face = faces[np.argmax(areas)]

        # Scale bbox back to original image size
        x1 = int(face.bbox[0] / scale)
        y1 = int(face.bbox[1] / scale)
        x2 = int(face.bbox[2] / scale)
        y2 = int(face.bbox[3] / scale)

        # Calculate width and height
        w = x2 - x1
        h = y2 - y1

        # Save original coordinates
        bb_path = img_path.parent / f"{img_path.stem}_BB.txt"
        with open(bb_path, "w") as f:
            f.write(f"{x1} {y1} {w} {h} {face.det_score:.7f}")

        return True

    except Exception as e:
        print(f"Error processing {img_path}: {str(e)}")
        return False


def process_dataset(dataset_path, dataset_name):
    """Process entire dataset"""
    print(f"\nProcessing {dataset_name}...")

    dataset_path = Path(dataset_path)

    # Find all images recursively
    image_files = []
    for ext in ["jpg", "png", "jpeg"]:
        image_files.extend(list(dataset_path.rglob(f"*.{ext}")))

    print(f"Found {len(image_files)} images")

    # Process each image
    success = 0
    for img_path in tqdm(image_files, desc="Detecting faces"):
        if process_image(img_path):
            success += 1

    print(f"Successfully processed {success}/{len(image_files)} images")
    return success


def main():
    print("Starting face redetection...")
    total_processed = 0

    # Process each dataset
    for name, path in DATASETS.items():
        if os.path.exists(path):
            # Remove old bbox files
            remove_old_bbox_files(path)

            # Process dataset with new detector
            total_processed += process_dataset(path, name)
        else:
            print(f"\nSkipping {name} - directory not found")

    print(f"\nTotal images processed: {total_processed}")


if __name__ == "__main__":
    main()
