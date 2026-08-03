"""
Download and prepare the AZH (Advancing the Zenith of Healthcare) wound
segmentation dataset for training.

The AZH dataset is hosted on GitHub:
https://github.com/uwm-bigdata/wound-segmentation

Usage:
    python download_azh.py --output_dir ./ml/data/raw

This script clones the repo (or you can do it manually) and reorganizes the
images/masks into a standard structure:

    ml/data/processed/
        train/
            images/
            masks/
        val/
            images/
            masks/
        test/
            images/
            masks/

NOTE: Review the dataset's license/usage terms before using it in any
public or commercial project.
"""

import argparse
import os
import shutil
import subprocess
import random

REPO_URL = "https://github.com/uwm-bigdata/wound-segmentation.git"


def clone_repo(target_dir: str):
    if os.path.exists(target_dir):
        print(f"Repo already present at {target_dir}, skipping clone.")
        return
    print(f"Cloning {REPO_URL} ...")
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, target_dir], check=True)


def collect_pairs(raw_dir: str):
    """
    Walks the cloned repo looking for image/mask pairs.
    The exact folder layout can vary by dataset version, so this performs a
    best-effort match on filenames between an 'images' and 'labels' (mask)
    directory. Inspect the cloned repo structure and adjust patterns as needed.
    """
    pairs = []
    for root, _dirs, files in os.walk(raw_dir):
        if os.path.basename(root).lower() in ("images", "img"):
            mask_root = root.replace("images", "labels").replace("img", "masks")
            for fname in files:
                img_path = os.path.join(root, fname)
                mask_path = os.path.join(mask_root, fname)
                if os.path.exists(mask_path):
                    pairs.append((img_path, mask_path))
    return pairs


def split_and_copy(pairs, output_dir: str, val_frac=0.15, test_frac=0.15, seed=42):
    random.seed(seed)
    random.shuffle(pairs)

    n = len(pairs)
    n_val = int(n * val_frac)
    n_test = int(n * test_frac)

    splits = {
        "val": pairs[:n_val],
        "test": pairs[n_val:n_val + n_test],
        "train": pairs[n_val + n_test:],
    }

    for split_name, split_pairs in splits.items():
        img_out = os.path.join(output_dir, split_name, "images")
        mask_out = os.path.join(output_dir, split_name, "masks")
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(mask_out, exist_ok=True)

        for img_path, mask_path in split_pairs:
            shutil.copy(img_path, os.path.join(img_out, os.path.basename(img_path)))
            shutil.copy(mask_path, os.path.join(mask_out, os.path.basename(mask_path)))

        print(f"{split_name}: {len(split_pairs)} pairs")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default="./ml/data/raw/wound-segmentation")
    parser.add_argument("--output_dir", default="./ml/data/processed")
    args = parser.parse_args()

    clone_repo(args.raw_dir)
    pairs = collect_pairs(args.raw_dir)
    print(f"Found {len(pairs)} image/mask pairs.")

    if not pairs:
        print(
            "No pairs found automatically. Open the cloned repo and check the "
            "actual folder names, then update collect_pairs() patterns."
        )
        return

    split_and_copy(pairs, args.output_dir)


if __name__ == "__main__":
    main()
