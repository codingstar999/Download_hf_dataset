"""
Dataset definitions for the validator cache system
"""

import os
import asyncio
import logging
from typing import List, Optional
from pathlib import Path
import huggingface_hub as hf_hub
import uuid

from dataset_type import Modality, MediaType, DatasetConfig
from download import download_files, list_hf_files
from filesystem import is_file_ready, wait_for_downloads_to_complete, is_zip_complete, is_parquet_complete


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


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
        # DatasetConfig(
        #     path="bitmind/bm-real",
        #     type=Modality.IMAGE,
        #     media_type=MediaType.REAL,
        # ),
        # DatasetConfig(
        #     path="bitmind/open-image-v7-256",
        #     type=Modality.IMAGE,
        #     media_type=MediaType.REAL,
        #     tags=["diverse"],
        # ),
        # DatasetConfig(
        #     path="bitmind/celeb-a-hq",
        #     type=Modality.IMAGE,
        #     media_type=MediaType.REAL,
        #     tags=["faces", "high-quality"],
        # ),
        # DatasetConfig(
        #     path="bitmind/ffhq-256",
        #     type=Modality.IMAGE,
        #     media_type=MediaType.REAL,
        #     tags=["faces", "high-quality"],
        # ),
        # DatasetConfig(
        #     path="bitmind/MS-COCO-unique-256",
        #     type=Modality.IMAGE,
        #     media_type=MediaType.REAL,
        #     tags=["diverse"],
        # ),
        # DatasetConfig(
        #     path="bitmind/AFHQ",
        #     type=Modality.IMAGE,
        #     media_type=MediaType.REAL,
        #     tags=["animals", "high-quality"],
        # ),
        # DatasetConfig(
        #     path="bitmind/lfw",
        #     type=Modality.IMAGE,
        #     media_type=MediaType.REAL,
        #     tags=["faces"],
        # ),
        # DatasetConfig(
        #     path="bitmind/caltech-256",
        #     type=Modality.IMAGE,
        #     media_type=MediaType.REAL,
        #     tags=["objects", "categorized"],
        # ),
        # DatasetConfig(
        #     path="bitmind/caltech-101",
        #     type=Modality.IMAGE,
        #     media_type=MediaType.REAL,
        #     tags=["objects", "categorized"],
        # ),
        # DatasetConfig(
        #     path="bitmind/dtd",
        #     type=Modality.IMAGE,
        #     media_type=MediaType.REAL,
        #     tags=["textures"],
        # ),
        # DatasetConfig(
        #     path="bitmind/idoc-mugshots-images",
        #     type=Modality.IMAGE,
        #     media_type=MediaType.REAL,
        #     tags=["faces"],
        # ),
        # # Synthetic image datasets
        # DatasetConfig(
        #     path="bitmind/JourneyDB",
        #     type=Modality.IMAGE,
        #     media_type=MediaType.SYNTHETIC,
        #     tags=["midjourney"],
        # ),
        # DatasetConfig(
        #     path="bitmind/GenImage_MidJourney",
        #     type=Modality.IMAGE,
        #     media_type=MediaType.SYNTHETIC,
        #     tags=["midjourney"],
        # ),
        # DatasetConfig(
        #     path="bitmind/bm-aura-imagegen",
        #     type=Modality.IMAGE,
        #     media_type=MediaType.SYNTHETIC,
        #     tags=["sora"],
        # ),
        # # Semisynthetic image datasets
        # DatasetConfig(
        #     path="bitmind/face-swap",
        #     type=Modality.IMAGE,
        #     media_type=MediaType.SEMISYNTHETIC,
        #     tags=["faces", "manipulated"],
        # ),
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
        # DatasetConfig(
        #     path="shangxd/imagenet-vidvrd",
        #     type=Modality.VIDEO,
        #     media_type=MediaType.REAL,
        #     tags=["diverse"],
        #     compressed_format="zip",
        # ),
        # DatasetConfig(
        #     path="nkp37/OpenVid-1M",
        #     type=Modality.VIDEO,
        #     media_type=MediaType.REAL,
        #     tags=["diverse", "large-zips"],
        #     compressed_format="zip",
        # ),
        # # Semisynthetic video datasets
        # DatasetConfig(
        #     path="bitmind/semisynthetic-video",
        #     type=Modality.VIDEO,
        #     media_type=MediaType.SEMISYNTHETIC,
        #     tags=["faces"],
        #     compressed_format="zip",
        # ),
    ]


async def download_dataset_files(dataset_config: DatasetConfig, output_dir: Path) -> List[Path]:
    """
    Download dataset files from Hugging Face using async downloads.
    
    Args:
        dataset_config: Dataset configuration
        output_dir: Directory to save downloaded files
        
    Returns:
        List of downloaded file paths
    """
    try:
        # Create dataset-specific directory
        dataset_dir = output_dir / dataset_config.path.replace("/", "_")
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        logging.info(f"Downloading dataset: {dataset_config.path}")
        
        # List files in the repository
        files = list_hf_files(dataset_config.path, repo_type="dataset")
        
        if not files:
            logging.warning(f"No files found in {dataset_config.path}")
            return []
        
        # Filter files based on dataset type and format
        if dataset_config.type == Modality.IMAGE:
            if dataset_config.compressed_format == "parquet":
                target_files = [f for f in files if f.endswith('.parquet')]
            else:
                # For images, look for common image formats
                target_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
        elif dataset_config.type == Modality.VIDEO:
            if dataset_config.compressed_format == "zip":
                target_files = [f for f in files if f.endswith('.zip')]
            else:
                target_files = [f for f in files if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        else:
            target_files = files
        
        if not target_files:
            logging.warning(f"No target files found for {dataset_config.path}")
            return []
        
        # Generate download URLs
        download_urls = []
        for file in target_files:
            url = f"https://huggingface.co/datasets/{dataset_config.path}/resolve/main/{file}"
            download_urls.append(url)
        
        logging.info(f"Found {len(download_urls)} files to download")
        
        # Download files asynchronously
        downloaded_files = await download_files(download_urls, dataset_dir)
        
        # Wait for downloads to complete and verify integrity
        if downloaded_files:
            await wait_for_downloads_to_complete(downloaded_files)
            
            # Verify file integrity
            valid_files = []
            for file_path in downloaded_files:
                if file_path.suffix.lower() == '.zip':
                    if is_zip_complete(file_path):
                        valid_files.append(file_path)
                    else:
                        logging.error(f"Invalid zip file: {file_path}")
                elif file_path.suffix.lower() == '.parquet':
                    if is_parquet_complete(file_path):
                        valid_files.append(file_path)
                    else:
                        logging.error(f"Invalid parquet file: {file_path}")
                else:
                    valid_files.append(file_path)
            
            logging.info(f"Successfully downloaded {len(valid_files)} valid files for {dataset_config.path}")
            return valid_files, dataset_dir
        
        return []
        
    except Exception as e:
        logging.error(f"Failed to download dataset {dataset_config.path}: {e}")
        return []


def extract_dataset_files(dataset_config: DatasetConfig, downloaded_files: List[Path], output_dir: Path):
    """
    Extract files from downloaded datasets and organize them in .cache/{modality}/{media_type}/
    """
    try:
        # Set up the .cache/{modality}/{media_type}/ directory
        modality = dataset_config.type.value  # 'image' or 'video'
        media_type = dataset_config.media_type.value  # 'real', 'synthetic', or 'semisynthetic'
        extract_dir = Path('./.cache') / modality / media_type / dataset_config.path.replace("/", "_")
        extract_dir.mkdir(parents=True, exist_ok=True)

        logging.info(f"Extracting files for {dataset_config.path} to {extract_dir}")

        for file_path in downloaded_files:
            if file_path.suffix.lower() == '.zip':
                extract_zip_file(file_path, extract_dir, dataset_config)
            elif file_path.suffix.lower() == '.parquet':
                extract_parquet_file(file_path, extract_dir, dataset_config)
            else:
                # Copy other files directly
                import shutil
                dest_path = extract_dir / file_path.name
                shutil.copy2(file_path, dest_path)

        logging.info(f"Extraction completed for {dataset_config.path}")

    except Exception as e:
        logging.error(f"Failed to extract {dataset_config.path}: {e}")


def extract_zip_file(zip_path: Path, extract_dir: Path, dataset_config: DatasetConfig):
    """Extract contents of a zip file, saving files with random names."""
    try:
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for member in zip_ref.namelist():
                # Only extract files (not directories)
                if not member.endswith('/'):
                    ext = Path(member).suffix or ''
                    random_name = f"{uuid.uuid4().hex}{ext}"
                    out_path = extract_dir / random_name
                    with zip_ref.open(member) as source, open(out_path, 'wb') as target:
                        target.write(source.read())
        logging.info(f"Extracted {zip_path.name}")
    except Exception as e:
        logging.error(f"Failed to extract {zip_path}: {e}")


def extract_parquet_file(parquet_path: Path, extract_dir: Path, dataset_config: DatasetConfig):
    """Extract images from a parquet file, saving with random names."""
    try:
        import pandas as pd
        import pyarrow.parquet as pq
        from PIL import Image
        import numpy as np
        
        # Read parquet file
        df = pq.read_table(parquet_path).to_pandas()
        
        # Look for image columns
        image_columns = [col for col in df.columns if 'image' in col.lower() or 'img' in col.lower()]
        
        if not image_columns:
            logging.warning(f"No image columns found in {parquet_path}")
            return
        
        # Extract images
        for i, row in df.iterrows():
            for col in image_columns:
                image_data = row[col]
                if image_data is not None:
                    try:
                        random_name = f"{uuid.uuid4().hex}.jpg"
                        image_path = extract_dir / random_name
                        # Handle different image formats
                        if hasattr(image_data, 'save'):
                            # PIL Image
                            image_data.save(image_path, "JPEG")
                        elif isinstance(image_data, bytes):
                            # Raw bytes
                            with open(image_path, 'wb') as f:
                                f.write(image_data)
                        elif isinstance(image_data, str):
                            # File path
                            import shutil
                            src_path = Path(image_data)
                            if src_path.exists():
                                ext = src_path.suffix or ''
                                random_name = f"{uuid.uuid4().hex}{ext}"
                                dst_path = extract_dir / random_name
                                shutil.copy2(src_path, dst_path)
                        elif isinstance(image_data, np.ndarray):
                            pil_image = Image.fromarray(image_data)
                            pil_image.save(image_path, "JPEG")
                    except Exception as e:
                        logging.warning(f"Failed to save image {i} from column {col}: {e}")
        
        logging.info(f"Extracted images from {parquet_path.name}")
        
    except Exception as e:
        logging.error(f"Failed to extract {parquet_path}: {e}")


async def download_and_extract_dataset(dataset_config: DatasetConfig, output_dir: Path):
    """
    Download and extract a dataset.
    
    Args:
        dataset_config: Dataset configuration
        output_dir: Base output directory
    """
    try:
        logging.info(f"Processing dataset: {dataset_config.path}")
        
        # Download files
        downloaded_files, dataset_dir = await download_dataset_files(dataset_config, output_dir)
        
        if downloaded_files:
            # Extract files
            extract_dataset_files(dataset_config, downloaded_files, output_dir)
            logging.info(f"Successfully processed {dataset_config.path}")
            import shutil
            shutil.rmtree(dataset_dir)
        else:
            logging.warning(f"No files downloaded for {dataset_config.path}")
            
    except Exception as e:
        logging.error(f"Failed to process {dataset_config.path}: {e}")


async def download_all_datasets(output_dir: Optional[Path] = None):
    """
    Download and extract all configured datasets.
    
    Args:
        output_dir: Output directory (defaults to current directory)
    """
    if output_dir is None:
        output_dir = Path.cwd()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all datasets
    image_datasets = get_image_datasets()
    video_datasets = get_video_datasets()
    all_datasets = image_datasets + video_datasets
    
    logging.info(f"Starting download of {len(all_datasets)} datasets")
    
    # Process datasets concurrently (with some limits to avoid overwhelming the server)
    semaphore = asyncio.Semaphore(3)  # Limit concurrent downloads
    
    async def process_with_semaphore(dataset):
        async with semaphore:
            await download_and_extract_dataset(dataset, output_dir)
    
    # Create tasks for all datasets
    tasks = [process_with_semaphore(dataset) for dataset in all_datasets]
    
    # Run all downloads
    await asyncio.gather(*tasks, return_exceptions=True)
    
    logging.info("All dataset downloads completed")


if __name__ == "__main__":
    # Print dataset configurations
    print("Image Datasets:")
    for ds in get_image_datasets():
        print(f"  {ds.path} ({ds.media_type.value})")
    
    print("\nVideo Datasets:")
    for ds in get_video_datasets():
        print(f"  {ds.path} ({ds.media_type.value})")
    
    # Run downloads
    print("\nStarting downloads...")
    asyncio.run(download_all_datasets())



