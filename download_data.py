"""
download_data.py
Downloads Quick, Draw! dataset .npy files for selected categories.
Each .npy file contains 28x28 grayscale bitmap images.
"""

import os
import urllib.request
import sys

# ── Categories we train on ──────────────────────────────────────────
CATEGORIES = [
    "cat", "dog", "house", "tree", "car",
    "sun", "moon", "fish", "bird", "flower",
    "sailboat", "castle", "mountain", "river", "star",
    "apple", "book", "hat", "umbrella", "key",
    "airplane", "alarm clock", "ant", "backpack", "banana", 
    "basketball", "beach", "bed", "bee", "belt", 
    "bench", "bicycle", "binoculars", "birthday cake", "blackberry", 
    "blueberry", "brain", "bread", "bridge", "broccoli", 
    "broom", "bus", "butterfly", "cactus", "cake", 
    "calculator", "calendar", "camera", "campfire", "candle"
]

BASE_URL = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def download_category(category: str) -> None:
    """Download a single category .npy file."""
    filename = f"{category}.npy"
    filepath = os.path.join(DATA_DIR, filename)

    if os.path.exists(filepath):
        print(f"  [OK] {category}.npy already exists, skipping.")
        return

    url = f"{BASE_URL}{category.replace(' ', '%20')}.npy"
    print(f"  [>>] Downloading {category}.npy ...", end=" ", flush=True)
    try:
        urllib.request.urlretrieve(url, filepath)
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"done ({size_mb:.1f} MB)")
    except Exception as e:
        print(f"FAILED: {e}")
        if os.path.exists(filepath):
            os.remove(filepath)


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    print("=" * 60)
    print("Quick, Draw! Dataset Downloader")
    print(f"Downloading {len(CATEGORIES)} categories to: {DATA_DIR}")
    print("=" * 60)

    for i, category in enumerate(CATEGORIES, 1):
        print(f"\n[{i}/{len(CATEGORIES)}] {category}")
        download_category(category)

    # Verify downloads
    downloaded = [f for f in os.listdir(DATA_DIR) if f.endswith(".npy")]
    print(f"\n{'=' * 60}")
    print(f"Download complete: {len(downloaded)}/{len(CATEGORIES)} files")
    print("=" * 60)

    if len(downloaded) < len(CATEGORIES):
        missing = [c for c in CATEGORIES if f"{c}.npy" not in downloaded]
        print(f"[!] Missing: {', '.join(missing)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
