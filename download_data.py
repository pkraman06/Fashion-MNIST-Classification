"""
download_data.py
Convenience script to pull the dataset from Kaggle into data/.

Setup:
  1. pip install kaggle
  2. Create a Kaggle API token (Kaggle account -> Settings -> Create New Token)
     and place kaggle.json at ~/.kaggle/kaggle.json  (chmod 600)
  3. Set config.KAGGLE_DATASET_SLUG to "<owner>/<dataset-name>", e.g.
     "masoudnickparvar/brain-tumor-mri-dataset"
  4. Run: python download_data.py

>>> Kaggle dataset link goes in config.py (config.KAGGLE_DATASET_URL) <<<
"""

import os
import subprocess
import sys

import config


def main():
    os.makedirs(config.DATA_DIR, exist_ok=True)

    if "<" in config.KAGGLE_DATASET_SLUG:
        print("Please set config.KAGGLE_DATASET_SLUG and config.KAGGLE_DATASET_URL "
              "to your chosen Kaggle dataset before running this script.")
        print(f"Example: {config.KAGGLE_DATASET_URL}")
        sys.exit(1)

    cmd = [
        "kaggle", "datasets", "download",
        "-d", config.KAGGLE_DATASET_SLUG,
        "-p", config.DATA_DIR,
        "--unzip",
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"Dataset downloaded and extracted to {config.DATA_DIR}")
    print("Expected subfolders: Training/ and Testing/, each with class subfolders.")


if __name__ == "__main__":
    main()
