import os
import numpy as np
import sys
import gc
from categories_split import CATEGORIES_SET1, CATEGORIES_SET2, CATEGORIES_SET3

# ── Configuration ───────────────────────────────────────────────────
SAMPLES_PER_CATEGORY = 5000
BASE_URL = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
TEMP_DIR = os.path.join(DATA_DIR, "temp_npy")

# Range to download: Header (~1KB) + 5000 samples * 784 bytes (~3.92MB)
# We take 4.5MB to be absolutely safe and handle variations.
DOWNLOAD_RANGE = "0-4500000" 

def download_category_turbo(category, retries=3):
    """Download only the first few KB of a category .npy file."""
    filename = f"{category}.npy"
    filepath = os.path.join(TEMP_DIR, filename)

    # Note: We overwrite partial files to be sure we have the Range-downloaded version
    # because if a previous run downloaded the FULL file, that's fine too.
    if os.path.exists(filepath):
        if os.path.getsize(filepath) > 4000000:
            return True

    url = f"{BASE_URL}{category.replace(' ', '%20')}.npy"
    for attempt in range(retries):
        try:
            print(f" (turbo att {attempt+1})...", end="", flush=True)
            # -r 0-300000: Download only first 300,001 bytes
            cmd = f'curl -L -s -r {DOWNLOAD_RANGE} -o "{filepath}" "{url}"'
            result = os.system(cmd)
            if result == 0 and os.path.exists(filepath) and os.path.getsize(filepath) > 100000:
                return True
        except Exception as e:
            if attempt == retries - 1:
                print(f"  [ERROR] Failed to download {category}: {e}")
            else:
                import time
                time.sleep(1)
    return False

def extract_samples(filepath, limit):
    """Extract sample pixels from a partial (or full) .npy file."""
    with open(filepath, "rb") as f:
        content = f.read()
        # Find where header ends (JSON dict followed by 0x0A)
        try:
            header_end = content.find(b'}') + 1
            # Skip padding (spaces or newlines)
            while header_end < len(content) and content[header_end] in [32, 10]: # 32=space, 10=newline
                header_end += 1
            
            # The data starts here. Grayscale bytes, 784 per sample.
            pixel_data = np.frombuffer(content[header_end:], dtype=np.uint8)
            num_found = len(pixel_data) // 784
            
            if num_found < 1:
                return None
            
            actual_count = min(num_found, limit)
            # Reshape to (N, 784) for stacking
            return pixel_data[:actual_count * 784].reshape(actual_count, 784)
        except:
            return None

def consolidate_set(categories, set_name):
    """Download and merge categories efficiently."""
    print(f"\n[*] Processing {set_name} ({len(categories)} categories)...")
    consolidated_data = []
    
    for i, category in enumerate(categories, 1):
        print(f"  [{i}/{len(categories)}] {category}...", end=" ", flush=True)
        if download_category_turbo(category):
            filepath = os.path.join(TEMP_DIR, f"{category}.npy")
            samples = extract_samples(filepath, SAMPLES_PER_CATEGORY)
            if samples is not None and samples.shape[0] > 0:
                consolidated_data.append(samples)
                print(f"done ({samples.shape[0]} samples)")
            else:
                print("FAILED (Extraction Error)")
                if os.path.exists(filepath): os.remove(filepath)
        else:
            print("FAILED (Download Error)")
        
        if i % 20 == 0: gc.collect()

    if consolidated_data:
        final_array = np.vstack(consolidated_data)
        output_path = os.path.join(DATA_DIR, f"{set_name}.npy")
        np.save(output_path, final_array)
        print(f"\n[OK] Saved {set_name}.npy: {final_array.shape[0]} samples total.")
    else:
        print(f"\n[!] No data collected for {set_name}")

def main():
    os.makedirs(TEMP_DIR, exist_ok=True)
    consolidate_set(CATEGORIES_SET1, "set1")
    consolidate_set(CATEGORIES_SET2, "set2")
    consolidate_set(CATEGORIES_SET3, "set3")
    print("\n" + "="*60)
    print("Turbo Data Prepping Complete!")
    print("="*60)

if __name__ == "__main__":
    main()
