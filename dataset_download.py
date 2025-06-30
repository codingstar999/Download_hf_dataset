"""
Dataset definitions for the validator cache system
"""

from typing import List
from datasets import load_dataset
import os
import zipfile
from pathlib import Path

from dataset_type import Modality, MediaType, DatasetConfig


def get_image_datasets() -> List[DatasetConfig]:
    """
    Get the list of image datasets used by the validator.

    Returns:
        List of image dataset configurations
    """
    return [
        # Real image datasets
        DatasetConfig(
            path="bitmind/bm-eidon-image",
            type=Modality.IMAGE,
            media_type=MediaType.REAL,
            tags=["frontier"],
        ),
        DatasetConfig(
            path="bitmind/bm-real",
            type=Modality.IMAGE,
            media_type=MediaType.REAL,
        ),
        DatasetConfig(
            path="bitmind/open-image-v7-256",
            type=Modality.IMAGE,
            media_type=MediaType.REAL,
            tags=["diverse"],
        ),
        DatasetConfig(
            path="bitmind/celeb-a-hq",
            type=Modality.IMAGE,
            media_type=MediaType.REAL,
            tags=["faces", "high-quality"],
        ),
        DatasetConfig(
            path="bitmind/ffhq-256",
            type=Modality.IMAGE,
            media_type=MediaType.REAL,
            tags=["faces", "high-quality"],
        ),
        DatasetConfig(
            path="bitmind/MS-COCO-unique-256",
            type=Modality.IMAGE,
            media_type=MediaType.REAL,
            tags=["diverse"],
        ),
        DatasetConfig(
            path="bitmind/AFHQ",
            type=Modality.IMAGE,
            media_type=MediaType.REAL,
            tags=["animals", "high-quality"],
        ),
        DatasetConfig(
            path="bitmind/lfw",
            type=Modality.IMAGE,
            media_type=MediaType.REAL,
            tags=["faces"],
        ),
        DatasetConfig(
            path="bitmind/caltech-256",
            type=Modality.IMAGE,
            media_type=MediaType.REAL,
            tags=["objects", "categorized"],
        ),
        DatasetConfig(
            path="bitmind/caltech-101",
            type=Modality.IMAGE,
            media_type=MediaType.REAL,
            tags=["objects", "categorized"],
        ),
        DatasetConfig(
            path="bitmind/dtd",
            type=Modality.IMAGE,
            media_type=MediaType.REAL,
            tags=["textures"],
        ),
        DatasetConfig(
            path="bitmind/idoc-mugshots-images",
            type=Modality.IMAGE,
            media_type=MediaType.REAL,
            tags=["faces"],
        ),
        # Synthetic image datasets
        DatasetConfig(
            path="bitmind/JourneyDB",
            type=Modality.IMAGE,
            media_type=MediaType.SYNTHETIC,
            tags=["midjourney"],
        ),
        DatasetConfig(
            path="bitmind/GenImage_MidJourney",
            type=Modality.IMAGE,
            media_type=MediaType.SYNTHETIC,
            tags=["midjourney"],
        ),
        DatasetConfig(
            path="bitmind/bm-aura-imagegen",
            type=Modality.IMAGE,
            media_type=MediaType.SYNTHETIC,
            tags=["sora"],
        ),
        # Semisynthetic image datasets
        DatasetConfig(
            path="bitmind/face-swap",
            type=Modality.IMAGE,
            media_type=MediaType.SEMISYNTHETIC,
            tags=["faces", "manipulated"],
        ),
    ]


def get_video_datasets() -> List[DatasetConfig]:
    """
    Get the list of video datasets used by the validator.
    """
    return [
        # Real video datasets
        DatasetConfig(
            path="bitmind/bm-eidon-video",
            type=Modality.VIDEO,
            media_type=MediaType.REAL,
            tags=["frontier"],
            compressed_format="zip",
        ),
        DatasetConfig(
            path="shangxd/imagenet-vidvrd",
            type=Modality.VIDEO,
            media_type=MediaType.REAL,
            tags=["diverse"],
            compressed_format="zip",
        ),
        DatasetConfig(
            path="nkp37/OpenVid-1M",
            type=Modality.VIDEO,
            media_type=MediaType.REAL,
            tags=["diverse", "large-zips"],
            compressed_format="zip",
        ),
        # Semisynthetic video datasets
        DatasetConfig(
            path="bitmind/semisynthetic-video",
            type=Modality.VIDEO,
            media_type=MediaType.SEMISYNTHETIC,
            tags=["faces"],
            compressed_format="zip",
        ),
    ]


def download_hf_dataset(dataset_config, cache_dir=None):
    print(f"Downloading {dataset_config.path} ...")
    try:
        ds = load_dataset(dataset_config.path, cache_dir=cache_dir)
        print(f"Downloaded: {dataset_config.path}")
        
        # If it's a real image dataset, try to extract files
        extract_dataset_files(ds, dataset_config, cache_dir)
            
    except Exception as e:
        print(f"Failed to download {dataset_config.path}: {e}")


def extract_dataset_files(dataset, dataset_config, cache_dir):
    """Extract files from downloaded dataset"""
    try:
        # Get the dataset directory
        dataset_dir = Path(cache_dir) / "downloads" / "extracted" / dataset_config.path.replace("/", "_")
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Extracting files to: {dataset_dir}")
        
        # Extract files from the dataset
        for split_name, split_data in dataset.items():
            split_dir = dataset_dir / split_name
            split_dir.mkdir(exist_ok=True)
            
            # Save images to the directory
            for i, example in enumerate(split_data):
                if 'image' in example:
                    image = example['image']
                    if hasattr(image, 'save'):
                        # PIL Image
                        image_path = split_dir / f"image_{i:06d}.jpg"
                        image.save(image_path)
                    elif isinstance(image, str):
                        # Image path
                        import shutil
                        src_path = Path(image)
                        if src_path.exists():
                            dst_path = split_dir / f"image_{i:06d}{src_path.suffix}"
                            shutil.copy2(src_path, dst_path)
        
        print(f"Extracted {dataset_config.path} to {dataset_dir}")
        
    except Exception as e:
        print(f"Failed to extract {dataset_config.path}: {e}")


if __name__ == "__main__":
    cache_dir = os.getcwd()  # current directory

    print("Image Datasets:")
    for ds in get_image_datasets():
        print(ds)
    print("\nVideo Datasets:")
    for ds in get_video_datasets():
        print(ds)
    print("\nDownloading image datasets:")
    for ds in get_image_datasets():
        download_hf_dataset(ds, cache_dir=cache_dir)
    print("\nDownloading video datasets:")
    for ds in get_video_datasets():
        download_hf_dataset(ds, cache_dir=cache_dir)



