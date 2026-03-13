"""
Download the trained model weights on demand.

Usage:
  python download_model.py --url <DIRECT_MODEL_URL>

Or use environment variable:
  MODEL_DOWNLOAD_URL=https://.../bert_anxiety_model.pt
  python download_model.py
"""
import argparse
import os
from pathlib import Path

import requests

from config import MODEL_DIR, MODEL_DOWNLOAD_TIMEOUT, MODEL_DOWNLOAD_URL, MODEL_PATH


def download_model(url: str, output_path: str = MODEL_PATH, timeout: int = MODEL_DOWNLOAD_TIMEOUT) -> str:
    """Download model weights from URL to output_path using a temporary file."""
    if not url:
        raise ValueError("Model URL is empty. Set MODEL_DOWNLOAD_URL or pass --url.")

    os.makedirs(MODEL_DIR, exist_ok=True)
    target = Path(output_path)
    temp_file = target.with_suffix(target.suffix + ".tmp")

    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", "0"))
        downloaded = 0

        with open(temp_file, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = (downloaded / total) * 100
                    print(f"Downloaded {downloaded / (1024 * 1024):.1f} MB / {total / (1024 * 1024):.1f} MB ({pct:.1f}%)")
                else:
                    print(f"Downloaded {downloaded / (1024 * 1024):.1f} MB")

    temp_file.replace(target)
    print(f"Model downloaded to: {target}")
    return str(target)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download model weights file.")
    parser.add_argument(
        "--url",
        type=str,
        default=MODEL_DOWNLOAD_URL,
        help="Direct URL to bert_anxiety_model.pt",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=MODEL_PATH,
        help="Destination path for model file",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=MODEL_DOWNLOAD_TIMEOUT,
        help="Request timeout in seconds",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        download_model(url=args.url, output_path=args.output, timeout=args.timeout)
    except Exception as exc:
        print(f"Failed to download model: {exc}")
        raise
