"""Evaluate generated images listed by generate_images.py."""

import argparse
import csv
import os
from pathlib import Path

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def load_image(path):
    image = np.load(path).astype(np.float32).squeeze()
    return image


def data_range(reference, generated):
    value = float(max(reference.max(), generated.max()) - min(reference.min(), generated.min()))
    return value if value > 0 else 1.0


def evaluate(manifest_path):
    results = []
    with open(manifest_path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            reference = load_image(row["input_path"])
            generated = load_image(row["generated_path"])
            value_range = data_range(reference, generated)
            results.append({
                **row,
                "mse": float(np.mean((reference - generated) ** 2)),
                "psnr": float(peak_signal_noise_ratio(reference, generated, data_range=value_range)),
                "ssim": float(structural_similarity(reference, generated, data_range=value_range)),
            })
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", help="manifest.csv produced by generate_images.py")
    parser.add_argument("--output")
    args = parser.parse_args()
    results = evaluate(args.manifest)
    if not results:
        raise ValueError("Manifest contains no generated images")

    output = Path(args.output) if args.output else Path(args.manifest).parent / "quality_metrics.csv"
    with open(output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    for metric in ("mse", "psnr", "ssim"):
        values = np.asarray([row[metric] for row in results])
        print(f"{metric.upper()}: {values.mean():.6f} ± {values.std():.6f}")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
