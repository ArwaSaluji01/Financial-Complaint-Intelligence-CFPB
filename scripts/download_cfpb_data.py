
import hashlib
import json
import sys
import zipfile

from datetime import datetime, timezone
from pathlib import Path

import requests
import yaml
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = PROJECT_ROOT / "config" / "data_config.yaml"


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def calculate_sha256(file_path):
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def download_archive(url, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        print("Archive already exists.")
        print("Skipping download.")
        return False

    temporary_path = destination.with_suffix(
        destination.suffix + ".part"
    )

    print("Downloading CFPB complaint archive...")
    print("Source:", url)

    try:
        response = requests.get(
            url,
            stream=True,
            timeout=(30, 180)
        )

        response.raise_for_status()

        total_size = int(
            response.headers.get("content-length", 0)
        )

        with open(temporary_path, "wb") as file:
            with tqdm(
                total=total_size if total_size else None,
                unit="B",
                unit_scale=True,
                desc="Downloading"
            ) as progress:

                for chunk in response.iter_content(
                    chunk_size=1024 * 1024
                ):
                    if chunk:
                        file.write(chunk)
                        progress.update(len(chunk))

        if not zipfile.is_zipfile(temporary_path):
            raise ValueError(
                "Downloaded file is not a valid ZIP archive."
            )

        temporary_path.replace(destination)

        print("\nDownload completed.")

        return True

    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()

        raise


def main():
    config = load_config()

    source = config["source"]
    paths = config["paths"]

    archive_path = PROJECT_ROOT / paths["raw_archive"]
    metadata_path = PROJECT_ROOT / paths["metadata"]

    try:
        downloaded = download_archive(
            source["archive_url"],
            archive_path
        )

        if not archive_path.exists():
            raise FileNotFoundError(
                f"Archive not found: {archive_path}"
            )

        file_hash = calculate_sha256(archive_path)

        with zipfile.ZipFile(archive_path, "r") as archive:
            archive_files = [
                item for item in archive.infolist()
                if not item.is_dir()
            ]

        metadata = {
            "dataset_name": source["name"],
            "archive_label": source["archive_label"],
            "source_url": source["archive_url"],
            "archive_page": source["archive_page"],
            "collection_timestamp_utc": (
                datetime.now(timezone.utc).isoformat()
            ),
            "downloaded_in_current_run": downloaded,
            "file_name": archive_path.name,
            "file_size_bytes": archive_path.stat().st_size,
            "sha256": file_hash,
            "archive_contents": [
                {
                    "name": item.filename,
                    "size_bytes": item.file_size
                }
                for item in archive_files
            ]
        }

        metadata_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(
            metadata_path,
            "w",
            encoding="utf-8"
        ) as file:
            json.dump(
                metadata,
                file,
                indent=4
            )

        print("\n" + "=" * 55)
        print("CFPB DATA ACQUISITION SUCCESSFUL")
        print("=" * 55)

        print("Archive:", archive_path)
        print("Size:", archive_path.stat().st_size, "bytes")
        print("SHA-256:", file_hash)

        print("\nFiles inside archive:")

        for item in archive_files:
            print(
                f"{item.filename} "
                f"({item.file_size:,} bytes)"
            )

        print("\nMetadata saved:", metadata_path)

    except Exception as error:
        print("\nDATA ACQUISITION FAILED")
        print("Reason:", error)
        sys.exit(1)


if __name__ == "__main__":
    main()
