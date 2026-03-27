"""Merge two EEG-ImageNet .pth dataset files into one."""

import pickle
import sys
from pathlib import Path

import torch


def load_pth(path: str) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError:
        return torch.load(path, map_location="cpu", weights_only=False)


def merge_pth(path_a: str, path_b: str, output_path: str) -> None:
    data_a = load_pth(path_a)
    data_b = load_pth(path_b)

    # Merge labels: preserve order from A, append any new labels from B
    labels_a = data_a["labels"]
    labels_b = data_b["labels"]
    labels_set = set(labels_a)
    merged_labels = list(labels_a) + [l for l in labels_b if l not in labels_set]

    # Merge images: same dedup logic
    images_a = data_a["images"]
    images_b = data_b["images"]
    images_set = set(images_a)
    merged_images = list(images_a) + [img for img in images_b if img not in images_set]

    # Merge dataset entries (plain concatenation — subjects/granularity are preserved per entry)
    merged_dataset = list(data_a["dataset"]) + list(data_b["dataset"])

    merged = {
        "labels": merged_labels,
        "images": merged_images,
        "dataset": merged_dataset,
    }

    torch.save(merged, output_path)
    print(f"Merged {len(data_a['dataset'])} + {len(data_b['dataset'])} "
          f"= {len(merged_dataset)} samples")
    print(f"Labels: {len(labels_a)} + {len(labels_b)} → {len(merged_labels)} unique")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python merge_pth.py <file_a.pth> <file_b.pth> <output.pth>")
        sys.exit(1)

    merge_pth(sys.argv[1], sys.argv[2], sys.argv[3])
