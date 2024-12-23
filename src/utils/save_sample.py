import os
import cv2
import random
from pathlib import Path

# Dataset paths
DATASETS = {
    "CelebA_Spoof": "./data/CelebA_Spoof_dataset",
    "CATI_FAS": "./data/CATI_FAS_dataset",
    "LCC_FASD": "./data/LCC_FASD_dataset",
    "NUAAA": "./data/NUAAA_dataset",
    "Zalo_AIC": "./data/Zalo_AIC_dataset",
}


def save_sample_images(dataset_path, num_samples=5):
    """
    Save sample images with and without bounding boxes
    Args:
        dataset_path: Path to dataset folder
        num_samples: Number of samples to save for each category (live/spoof)
    """
    print(f"\nSaving sample images for {Path(dataset_path).name}...")

    # Create output directory
    output_dir = Path("./output") / f"{Path(dataset_path).name}_sample"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process live and spoof folders
    for folder in ["live", "spoof", "live/live", "spoof/spoof"]:
        folder_path = Path(dataset_path) / folder
        if not folder_path.exists():
            continue

        # Get all image files
        all_images = []
        for ext in [".jpg", ".png"]:
            all_images.extend(list(folder_path.glob(f"*{ext}")))

        if not all_images:
            continue

        # Separate images with and without bbox
        with_bbox = []
        without_bbox = []

        for img_path in all_images:
            bbox_path = img_path.parent / f"{img_path.stem}_BB.txt"
            if bbox_path.exists():
                with_bbox.append(img_path)
            else:
                without_bbox.append(img_path)

        print(f"\n{folder} stats:")
        print(f"Total images: {len(all_images)}")
        print(f"With bbox: {len(with_bbox)}")
        print(f"Without bbox: {len(without_bbox)}")

        # Select random samples from each group
        selected_with_bbox = (
            random.sample(with_bbox, min(num_samples, len(with_bbox)))
            if with_bbox
            else []
        )
        selected_without_bbox = (
            random.sample(without_bbox, min(num_samples, len(without_bbox)))
            if without_bbox
            else []
        )

        # Process samples with bbox
        for i, img_path in enumerate(selected_with_bbox):
            try:
                # Read image
                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                # Save original image
                output_path = output_dir / f"{folder}_{i+1}_original.jpg"
                cv2.imwrite(str(output_path), img)

                # Read and draw bbox
                bbox_path = img_path.parent / f"{img_path.stem}_BB.txt"
                with open(bbox_path) as f:
                    x, y, w, h, conf = map(float, f.read().strip().split())

                # Draw bbox on a copy of the image
                img_with_bbox = img.copy()
                cv2.rectangle(
                    img_with_bbox,
                    (int(x), int(y)),
                    (int(x + w), int(y + h)),
                    (0, 255, 0),
                    2,
                )

                # Add label with confidence score
                label = f"{folder} ({conf:.3f})"
                cv2.putText(
                    img_with_bbox,
                    label,
                    (int(x), int(y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

                # Save image with bbox
                output_path = output_dir / f"{folder}_{i+1}_with_bbox.jpg"
                cv2.imwrite(str(output_path), img_with_bbox)

            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
                continue

        # Process samples without bbox
        for i, img_path in enumerate(selected_without_bbox):
            try:
                # Read and save image
                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                output_path = output_dir / f"{folder}_{i+1}_no_bbox.jpg"
                cv2.imwrite(str(output_path), img)

            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
                continue

    print(f"Samples saved to {output_dir}")


def main():
    """Process each dataset and save sample images"""
    for name, path in DATASETS.items():
        if not os.path.exists(path):
            print(f"\nSkipping {name} - directory not found")
            continue

        save_sample_images(path)
        print(f"\nFinished processing {name}")


if __name__ == "__main__":
    main()
