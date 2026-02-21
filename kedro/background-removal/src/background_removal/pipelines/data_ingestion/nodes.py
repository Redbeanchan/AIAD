"""
This is a boilerplate pipeline 'data_ingestion'
generated using Kedro 0.19.14
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict
from zipfile import ZipFile

import gdown


def download_and_extract_many_from_gdrive(
    files: Dict[str, str],
    output_root: str,
    delete_zip: bool = True,
) -> Dict[str, str]:
    """
    Download multiple zip files from Google Drive (by file_id) and extract each into:
      <output_root>/<folder_name>/

    Returns:
      dict mapping folder_name -> extracted folder path
    """
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)

    extracted_dirs: Dict[str, str] = {}

    for folder_name, file_id in files.items():
        out_path = root / folder_name
        out_path.mkdir(parents=True, exist_ok=True)

        zip_path = out_path / f"{folder_name}.zip"
        url = f"https://drive.google.com/uc?id={file_id}"

        # Download
        gdown.download(url, str(zip_path), quiet=False)

        # Extract
        with ZipFile(zip_path, "r") as z:
            z.extractall(out_path)

        # Cleanup
        if delete_zip and zip_path.exists():
            zip_path.unlink()

        extracted_dirs[folder_name] = str(out_path)

    return extracted_dirs  