"""
Download the MedSAM checkpoint.

MedSAM's weights are distributed via Google Drive/Hugging Face by the authors
(bowang-lab/MedSAM), not a stable direct-download URL, so this script takes
the file ID/URL as an argument rather than hardcoding one that could go stale
or be wrong.

Steps:
1. Go to https://github.com/bowang-lab/MedSAM and find the current checkpoint
   link in their README (look for "medsam_vit_b.pth").
2. If it's a Google Drive link, grab the file ID from the URL
   (drive.google.com/file/d/<FILE_ID>/view) and run:

     python download_medsam_checkpoint.py --gdrive_id <FILE_ID>

   Or if it's a direct HTTPS link (e.g. Hugging Face):

     python download_medsam_checkpoint.py --url <DIRECT_URL>

Saves to ml/checkpoints/medsam_vit_b.pth by default.
"""

import argparse
import os


def download_from_gdrive(file_id: str, output_path: str):
    import gdown
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output_path, quiet=False)


def download_from_url(url: str, output_path: str):
    import urllib.request
    urllib.request.urlretrieve(url, output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gdrive_id", default=None, help="Google Drive file ID for medsam_vit_b.pth")
    parser.add_argument("--url", default=None, help="Direct HTTPS URL to the checkpoint")
    parser.add_argument("--output", default="./ml/checkpoints/medsam_vit_b.pth")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    if args.gdrive_id:
        download_from_gdrive(args.gdrive_id, args.output)
    elif args.url:
        download_from_url(args.url, args.output)
    else:
        print(
            "Provide --gdrive_id or --url. See the docstring at the top of this "
            "file for how to find the current MedSAM checkpoint link."
        )
        return

    print(f"Saved checkpoint to {args.output}")


if __name__ == "__main__":
    main()
