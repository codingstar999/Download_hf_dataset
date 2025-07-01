import os
import csv
from pathlib import Path

def write_image_paths_csv(root_dir: Path, exts, csv_path: Path):
    """Write all image file paths with given extensions under root_dir to csv_path (absolute paths)."""
    paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if any(fname.lower().endswith(ext) for ext in exts):
                full_path = (Path(dirpath) / fname).resolve()
                paths.append([str(full_path)])
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_path'])
        writer.writerows(paths)
    print(f"Wrote {len(paths)} image paths to {csv_path}")

def write_semisynthetic_image_label_csv(root_dir: Path, exts, csv_path: Path):
    """Write image and label (.npy) file paths for semisynthetic images (absolute paths)."""
    rows = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if any(fname.lower().endswith(ext) for ext in exts):
                image_path = (Path(dirpath) / fname).resolve()
                label_path = image_path.with_name(image_path.stem + '_mask.npy')
                # If label exists, use it; else, leave blank
                if label_path.exists():
                    rows.append([str(image_path), str(label_path)])
                else:
                    rows.append([str(image_path), ''])
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_path', 'label_path'])
        writer.writerows(rows)
    print(f"Wrote {len(rows)} image-label pairs to {csv_path}")

if __name__ == "__main__":
    base_cache = Path.cwd() / '.cache'
    real_root = base_cache / 'image' / 'real'
    semisynthetic_root = base_cache / 'image' / 'semisynthetic'
    synthetic_root = base_cache / 'image' / 'synthetic'
    image_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.gif']

    write_image_paths_csv(real_root, image_exts, base_cache / 'image' / 'real_image_paths.csv')
    write_image_paths_csv(synthetic_root, image_exts, base_cache / 'image' / 'synthetic_image_paths.csv')
    write_image_paths_csv(semisynthetic_root, image_exts, base_cache / 'image' / 'semisynthetic_image_paths.csv')
    write_semisynthetic_image_label_csv(semisynthetic_root, image_exts, base_cache / 'image' / 'semisynthetic_image_label_paths.csv')

