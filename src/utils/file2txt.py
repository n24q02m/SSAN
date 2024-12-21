import os
from pathlib import Path
from dotenv import load_dotenv
import shutil


def load_export_location():
    """Load export location from .env file"""
    load_dotenv()
    export_location = os.getenv("FILE_EXPORTED_LOCATION")
    if not export_location:
        raise ValueError("FILE_EXPORTED_LOCATION not found in .env file")
    return export_location


def should_convert(file_path):
    """Check if file should be converted based on extension"""
    return file_path.suffix.lower() in [".py", ".md"]


def convert_to_txt(source_path, export_root):
    """Convert file to .txt and save to export location while preserving structure"""
    # Get relative path to maintain directory structure
    rel_path = os.path.relpath(source_path)

    # Create destination path with .txt extension
    dest_path = os.path.join(export_root, f"{rel_path}.txt")

    # Create destination directory if it doesn't exist
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    # Copy file with new extension
    shutil.copy2(source_path, dest_path)
    print(f"Converted: {rel_path} -> {dest_path}")


def process_directory(dir_path, export_root):
    """Process directory recursively"""
    ignore_folder = [".pytest_cache", ".git"]
    
    for root, _, files in os.walk(dir_path):
        if any(folder in root for folder in ignore_folder):
            continue

        for file in files:
            file_path = Path(os.path.join(root, file))
            if should_convert(file_path):
                convert_to_txt(file_path, export_root)


def clean_export_directory(export_root):
    """Clean all files in export directory"""
    if os.path.exists(export_root):
        shutil.rmtree(export_root)
        print(f"Cleaned export directory: {export_root}")
    os.makedirs(export_root)
    print(f"Created new export directory: {export_root}")


def main():
    try:
        # Get current directory
        current_dir = os.path.abspath(".")

        # Get export location from .env
        export_root = load_export_location()

        # Clean export directory
        clean_export_directory(export_root)

        # Process directory
        process_directory(current_dir, export_root)

        print(f"\nFiles have been exported to: {export_root}")

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
