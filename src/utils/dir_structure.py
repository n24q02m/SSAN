import os


def count_dirs_and_files(path):
    """Count number of directories and files separately in a path"""
    try:
        items = list(os.scandir(path))
        dirs = sum(1 for item in items if item.is_dir())
        files = sum(1 for item in items if item.is_file())
        return dirs, files
    except PermissionError:
        return 0, 0


def should_summarize(path):
    """Check if a directory should be summarized based on number of dirs/files"""
    dirs, files = count_dirs_and_files(path)
    return dirs > 10 or files > 10


def write_tree(path, output_file, prefix="", is_last=True):
    """Write directory tree structure to file"""
    if ".git" in str(path):
        return

    # Get directory name
    name = os.path.basename(path)

    # Skip if empty name (happens for root directory)
    if not name:
        name = str(path)

    # Write current directory/file
    if is_last:
        output_file.write(f"{prefix}└── {name}/\n")
        new_prefix = f"{prefix}    "
    else:
        output_file.write(f"{prefix}├── {name}/\n")
        new_prefix = f"{prefix}│   "

    # If it's a directory, process its contents
    if os.path.isdir(path):
        try:
            # Get and sort contents, separating dirs and files
            contents = list(os.scandir(path))
            dirs = sorted(
                [x for x in contents if x.is_dir()], key=lambda x: x.name.lower()
            )
            files = sorted(
                [x for x in contents if x.is_file()], key=lambda x: x.name.lower()
            )

            # Check if we should summarize
            if should_summarize(path):
                # Show first 10 dirs
                visible_dirs = dirs[:10]
                remaining_dirs = len(dirs) - 10 if len(dirs) > 10 else 0

                # Show first 10 files
                visible_files = files[:10]
                remaining_files = len(files) - 10 if len(files) > 10 else 0

                # Process visible items
                visible_items = visible_dirs + visible_files
                for i, item in enumerate(visible_items):
                    is_last_visible = (
                        i == len(visible_items) - 1
                        and remaining_dirs == 0
                        and remaining_files == 0
                    )
                    write_tree(item.path, output_file, new_prefix, is_last_visible)

                # Add summary for dirs if needed
                if remaining_dirs > 0:
                    output_file.write(f"{new_prefix}... ({remaining_dirs} more dirs)\n")

                # Add summary for files if needed
                if remaining_files > 0:
                    output_file.write(
                        f"{new_prefix}... ({remaining_files} more files)\n"
                    )
            else:
                # Show all items if both dirs and files are <= 10
                all_items = dirs + files
                for i, item in enumerate(all_items):
                    is_last = i == len(all_items) - 1
                    write_tree(item.path, output_file, new_prefix, is_last)

        except PermissionError:
            output_file.write(f"{new_prefix}[Permission Denied]\n")


def main():
    # Create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)

    # Delete old structure file if exists
    structure_file = "output/dir_structure.txt"
    if os.path.exists(structure_file):
        os.remove(structure_file)
        print(f"Deleted old file: {structure_file}")

    # Open output file
    with open(structure_file, "w", encoding="utf-8") as f:
        # Get root directory (current directory)
        root_path = os.path.abspath(".")

        # Write tree structure
        write_tree(root_path, f)

    print(f"Created new structure file: {structure_file}")


if __name__ == "__main__":
    main()
