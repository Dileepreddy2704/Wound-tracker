"""
Fine-tune a segmentation model (SegFormer, or MedSAM/SAM2 with a prompt-free
adapter) on the processed wound dataset.

This is a starting skeleton — fill in the model-specific pieces once you've
picked SegFormer vs. MedSAM as the primary approach:

  - SegFormer: straightforward encoder-decoder fine-tuning via
    `transformers.SegformerForSemanticSegmentation`, binary wound/background
    classes. Easier to train from scratch on a small dataset.
  - MedSAM/SAM2: prompt-based (point/box) segmentation foundation model.
    Stronger zero-shot, but needs a prompt-generation strategy (e.g. auto
    bounding box from a coarse detector) since you won't have a clinician
    clicking points at inference time.

Recommended starting point given dataset size: fine-tune SegFormer first to
get an end-to-end working pipeline, evaluate MedSAM as a stretch goal.
"""

import argparse
import os

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np


class WoundSegDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.filenames = sorted(os.listdir(images_dir))
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        image = Image.open(os.path.join(self.images_dir, fname)).convert("RGB")
        mask = Image.open(os.path.join(self.masks_dir, fname)).convert("L")

        image = np.array(image)
        mask = (np.array(mask) > 127).astype(np.float32)  # binarize

        if self.transform:
            image, mask = self.transform(image, mask)

        return torch.from_numpy(image).permute(2, 0, 1).float() / 255.0, torch.from_numpy(mask)


def train(args):
    train_ds = WoundSegDataset(
        os.path.join(args.data_dir, "train", "images"),
        os.path.join(args.data_dir, "train", "masks"),
    )
    val_ds = WoundSegDataset(
        os.path.join(args.data_dir, "val", "images"),
        os.path.join(args.data_dir, "val", "masks"),
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    # TODO: instantiate model, e.g.:
    # from transformers import SegformerForSemanticSegmentation
    # model = SegformerForSemanticSegmentation.from_pretrained(
    #     "nvidia/segformer-b0-finetuned-ade-512-512",
    #     num_labels=2,
    #     ignore_mismatched_sizes=True,
    # )

    # TODO: training loop — optimizer, loss (Dice + BCE is a good combo for
    # segmentation with class imbalance since wound area is usually small
    # relative to the full image), checkpoint saving.

    print("Model/training loop not yet implemented — wire in SegFormer or MedSAM next.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./ml/data/processed")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    train(args)
