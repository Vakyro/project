"""
Zenodo Downloader for CLIPZyme Artifacts

Downloads model checkpoints and datasets from Zenodo repository.

CLIPZyme Zenodo Record: https://zenodo.org/records/15161343
- clipzyme_model.zip (2.4 GB) - Pre-trained model checkpoint
- clipzyme_data.zip (1.3 GB) - Training and evaluation datasets
- reaction_rule_split.p (1.9 kB) - Reaction rule splits

DOI: 10.5281/zenodo.15161343
"""

import requests
import hashlib
from pathlib import Path
from typing import Optional, Dict, List
from tqdm import tqdm
import zipfile
import logging

logger = logging.getLogger(__name__)


class ZenodoDownloader:
    """
    Download files from Zenodo repository.

    Supports:
    - Resume interrupted downloads
    - Checksum verification
    - Progress tracking
    - Automatic extraction
    """

    # CLIPZyme official Zenodo record
    CLIPZYME_RECORD_ID = "15161343"
    ZENODO_API_BASE = "https://zenodo.org/api/records"

    # Known file metadata
    CLIPZYME_FILES = {
        'model': {
            'filename': 'clipzyme_model.zip',
            'size_gb': 2.4,
            'description': 'Pre-trained CLIPZyme model checkpoint',
        },
        'data': {
            'filename': 'clipzyme_data.zip',
            'size_gb': 1.3,
            'description': 'Training and evaluation datasets',
        },
        'splits': {
            'filename': 'reaction_rule_split.p',
            'size_kb': 1.9,
            'description': 'Reaction rule train/test splits',
        }
    }

    def __init__(self, output_dir: str = "data/checkpoints"):
        """
        Initialize downloader.

        Args:
            output_dir: Directory to save downloaded files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_record_metadata(self, record_id: str = None) -> Dict:
        """
        Get metadata for a Zenodo record.

        Args:
            record_id: Zenodo record ID (default: CLIPZyme record)

        Returns:
            Dictionary with record metadata
        """
        record_id = record_id or self.CLIPZYME_RECORD_ID
        url = f"{self.ZENODO_API_BASE}/{record_id}"

        logger.info(f"Fetching metadata from {url}")
        response = requests.get(url)
        response.raise_for_status()

        return response.json()

    def list_files(self, record_id: str = None) -> List[Dict]:
        """
        List all files in a Zenodo record.

        Args:
            record_id: Zenodo record ID

        Returns:
            List of file metadata dictionaries
        """
        metadata = self.get_record_metadata(record_id)
        files = metadata.get('files', [])

        logger.info(f"Found {len(files)} files in record")
        for file_info in files:
            size_mb = file_info['size'] / (1024 ** 2)
            logger.info(f"  - {file_info['key']}: {size_mb:.2f} MB")

        return files

    def download_file(
        self,
        url: str,
        output_path: Path,
        checksum: Optional[str] = None,
        resume: bool = True
    ) -> Path:
        """
        Download a file with progress bar and resume support.

        Args:
            url: Download URL
            output_path: Path to save file
            checksum: Optional MD5 checksum for verification
            resume: If True, resume interrupted downloads

        Returns:
            Path to downloaded file
        """
        # Check if file already exists and is complete
        if output_path.exists():
            if checksum:
                if self._verify_checksum(output_path, checksum):
                    logger.info(f"File already downloaded and verified: {output_path}")
                    return output_path
                else:
                    logger.warning(f"Existing file failed checksum, re-downloading")
                    output_path.unlink()
            elif resume:
                logger.info(f"File exists, will attempt resume: {output_path}")

        # Prepare headers for resume
        headers = {}
        initial_pos = 0
        if resume and output_path.exists():
            initial_pos = output_path.stat().st_size
            headers['Range'] = f'bytes={initial_pos}-'

        # Start download
        logger.info(f"Downloading from {url}")
        response = requests.get(url, headers=headers, stream=True)

        # Check if resume is supported
        if resume and response.status_code == 206:
            mode = 'ab'
            logger.info(f"Resuming download from byte {initial_pos}")
        elif response.status_code == 200:
            mode = 'wb'
            initial_pos = 0
        else:
            response.raise_for_status()
            mode = 'wb'

        # Get total size
        total_size = int(response.headers.get('content-length', 0)) + initial_pos

        # Download with progress bar
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, mode) as f:
            with tqdm(
                total=total_size,
                initial=initial_pos,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
                desc=output_path.name
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        logger.info(f"Download complete: {output_path}")

        # Verify checksum if provided
        if checksum:
            if self._verify_checksum(output_path, checksum):
                logger.info("✓ Checksum verified")
            else:
                raise ValueError("Checksum verification failed!")

        return output_path

    def _verify_checksum(self, file_path: Path, expected_md5: str) -> bool:
        """Verify MD5 checksum of a file."""
        logger.info(f"Verifying checksum for {file_path.name}...")

        md5_hash = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                md5_hash.update(chunk)

        actual_md5 = md5_hash.hexdigest()
        return actual_md5 == expected_md5

    def download_clipzyme_file(
        self,
        file_key: str,
        extract: bool = True,
        record_id: str = None
    ) -> Path:
        """
        Download a CLIPZyme file by key.

        Args:
            file_key: One of 'model', 'data', or 'splits'
            extract: If True, extract zip files after download
            record_id: Zenodo record ID (default: official CLIPZyme)

        Returns:
            Path to downloaded (and possibly extracted) file
        """
        if file_key not in self.CLIPZYME_FILES:
            raise ValueError(
                f"Unknown file key: {file_key}. "
                f"Must be one of {list(self.CLIPZYME_FILES.keys())}"
            )

        file_info = self.CLIPZYME_FILES[file_key]
        logger.info(f"Downloading {file_key}: {file_info['description']}")

        # Get download URL from Zenodo API
        metadata = self.get_record_metadata(record_id)
        files = metadata.get('files', [])

        target_file = None
        for f in files:
            if f['key'] == file_info['filename']:
                target_file = f
                break

        if not target_file:
            raise ValueError(
                f"File {file_info['filename']} not found in Zenodo record"
            )

        # Download
        download_url = target_file['links']['self']
        output_path = self.output_dir / file_info['filename']
        checksum = target_file.get('checksum', '').replace('md5:', '')

        downloaded_path = self.download_file(
            url=download_url,
            output_path=output_path,
            checksum=checksum if checksum else None,
            resume=True
        )

        # Extract if zip file
        if extract and downloaded_path.suffix == '.zip':
            logger.info(f"Extracting {downloaded_path.name}...")
            extract_dir = downloaded_path.parent / downloaded_path.stem
            extract_dir.mkdir(exist_ok=True)

            with zipfile.ZipFile(downloaded_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)

            logger.info(f"✓ Extracted to {extract_dir}")
            return extract_dir

        return downloaded_path

    def download_all(self, extract: bool = True) -> Dict[str, Path]:
        """
        Download all CLIPZyme files.

        Args:
            extract: If True, extract zip files

        Returns:
            Dictionary mapping file keys to downloaded paths
        """
        logger.info("Downloading all CLIPZyme files from Zenodo...")

        results = {}
        for file_key in self.CLIPZYME_FILES.keys():
            try:
                path = self.download_clipzyme_file(file_key, extract=extract)
                results[file_key] = path
            except Exception as e:
                logger.error(f"Failed to download {file_key}: {e}")
                results[file_key] = None

        logger.info("Download complete!")
        return results


def download_clipzyme_checkpoint(
    output_dir: str = "data/checkpoints",
    extract: bool = True
) -> Path:
    """
    Convenience function to download CLIPZyme model checkpoint.

    Args:
        output_dir: Directory to save checkpoint
        extract: If True, extract the zip file

    Returns:
        Path to downloaded checkpoint directory
    """
    downloader = ZenodoDownloader(output_dir=output_dir)
    checkpoint_path = downloader.download_clipzyme_file('model', extract=extract)

    logger.info(f"CLIPZyme checkpoint downloaded to: {checkpoint_path}")
    return checkpoint_path


def download_clipzyme_data(
    output_dir: str = "data/datasets",
    extract: bool = True
) -> Path:
    """
    Convenience function to download CLIPZyme datasets.

    Args:
        output_dir: Directory to save data
        extract: If True, extract the zip file

    Returns:
        Path to downloaded data directory
    """
    downloader = ZenodoDownloader(output_dir=output_dir)
    data_path = downloader.download_clipzyme_file('data', extract=extract)

    logger.info(f"CLIPZyme data downloaded to: {data_path}")
    return data_path


def download_clipzyme_splits(
    output_dir: str = "data/datasets"
) -> Path:
    """
    Convenience function to download reaction rule splits.

    Args:
        output_dir: Directory to save splits

    Returns:
        Path to downloaded splits file
    """
    downloader = ZenodoDownloader(output_dir=output_dir)
    splits_path = downloader.download_clipzyme_file('splits', extract=False)

    logger.info(f"Reaction splits downloaded to: {splits_path}")
    return splits_path


__all__ = [
    'ZenodoDownloader',
    'download_clipzyme_checkpoint',
    'download_clipzyme_data',
    'download_clipzyme_splits',
]
