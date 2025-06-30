import os
import csv
from pathlib import Path

def write_paths_csv(root_dir: Path, exts, csv_path: Path):
    """Write all file paths with given extensions under root_dir to csv_path."""
    paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if any(fname.lower().endswith(ext) for ext in exts):
                full_path = Path(dirpath) / fname
                rel_path = full_path.relative_to(root_dir.parent)
                paths.append([str(rel_path)])
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['path'])
        writer.writerows(paths)
    print(f"Wrote {len(paths)} paths to {csv_path}")

if __name__ == "__main__":
    base_cache = Path.cwd() / '.cache'
    image_root = base_cache / 'image'
    video_root = base_cache / 'video'
    write_paths_csv(image_root, exts=['.jpg', '.jpeg', '.png', '.bmp', '.tiff'], csv_path=image_root / 'image_paths.csv')
    write_paths_csv(video_root, exts=['.mp4', '.avi', '.mov', '.mkv'], csv_path=video_root / 'video_paths.csv')
