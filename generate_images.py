"""Generate reconstructed MRI images from a trained RGDM or VST checkpoint."""

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
import torchvision.utils as vutils

from experiment_registry import load_experiment_components
from helpers import defaultdict_from_json, load_checkpoint
from UNet import UNetModel

PROJECT_ROOT = Path(__file__).parent


def load_args(arg_name, dataset_override=None, method_override=None):
    name = str(arg_name)
    if name.isnumeric():
        name = f"args{name}.json"
    elif name.startswith("args") and not name.endswith(".json"):
        name = f"{name}.json"

    with open(PROJECT_ROOT / "test_args" / name, encoding="utf-8") as handle:
        args = defaultdict_from_json(json.load(handle))
    args["arg_num"] = name[4:-5]
    if dataset_override:
        args["dataset"] = dataset_override
    if method_override:
        args["method"] = method_override
    return args


def build_model(args, checkpoint, device):
    channels = args["channels"] if args["channels"] != "" else 1
    model = UNetModel(
        args["img_size"][0], args["base_channels"],
        channel_mults=args["channel_mults"], dropout=args["dropout"],
        n_heads=args["num_heads"], n_head_channels=args["num_head_channels"],
        in_channels=channels,
    ).to(device)
    state = checkpoint.get("ema") or checkpoint.get("model_state_dict") or checkpoint.get("unet")
    if state is None:
        raise KeyError("Checkpoint does not contain ema, model_state_dict, or unet weights")
    model.load_state_dict(state)
    model.eval()
    return model, channels


def generate(args, checkpoint, output_dir, count, t_distance, device):
    dataset_module, diffusion_module = load_experiment_components(args)
    _, test_dataset = dataset_module.init_datasets(PROJECT_ROOT, args)
    loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args["Batch_Size"], shuffle=False,
        num_workers=0, drop_last=False,
    )
    model, channels = build_model(args, checkpoint, device)
    diffusion = diffusion_module.GaussianDiffusionModel(
        args["img_size"], diffusion_module.get_beta_schedule(args["T"], args["beta_schedule"]),
        loss_weight=args["loss_weight"], loss_type=args["loss-type"],
        noise=args["noise_fn"], img_channels=channels, domain=args["method"],
    )

    input_dir = os.path.join(output_dir, "input")
    generated_dir = os.path.join(output_dir, "generated")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(generated_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, "manifest.csv")

    rows = []
    generated_count = 0
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            reconstructions = diffusion.forward_backward(
                model, images, see_whole_sequence=None, t_distance=t_distance,
            )
            names = batch.get("filenames", [str(i) for i in range(images.shape[0])])
            for index in range(images.shape[0]):
                if generated_count >= count:
                    break
                sample_id = str(names[index]).replace(os.sep, "_")
                stem = f"{generated_count:05d}_{sample_id}"
                input_npy = os.path.join(input_dir, f"{stem}.npy")
                generated_npy = os.path.join(generated_dir, f"{stem}.npy")
                input_tensor = images[index].detach().cpu()
                generated_tensor = reconstructions[index].detach().cpu()
                np.save(input_npy, input_tensor.numpy())
                np.save(generated_npy, generated_tensor.numpy())
                vutils.save_image(input_tensor, os.path.join(input_dir, f"{stem}.png"), normalize=True)
                vutils.save_image(generated_tensor, os.path.join(generated_dir, f"{stem}.png"), normalize=True)
                rows.append({
                    "sample_id": sample_id,
                    "dataset": args["dataset"],
                    "method": args["method"],
                    "input_path": input_npy,
                    "generated_path": generated_npy,
                })
                generated_count += 1
            if generated_count >= count:
                break

    with open(manifest_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys() if rows else [
            "sample_id", "dataset", "method", "input_path", "generated_path"
        ])
        writer.writeheader()
        writer.writerows(rows)
    return manifest_path, generated_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("args", help="Experiment number, argsNUM, or argsNUM.json")
    parser.add_argument("--dataset", choices=["t1", "t2", "flair", "nfbs", "knee"])
    parser.add_argument("--method", choices=["rgdm", "vst"])
    parser.add_argument("--checkpoint", help="Override the configured final checkpoint")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--t-distance", type=int, default=500)
    parser.add_argument("--output-dir")
    options = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = load_args(options.args, options.dataset, options.method)
    load_experiment_components(args)  # validate and normalise values
    checkpoint = (
        torch.load(options.checkpoint, map_location=device, weights_only=False)
        if options.checkpoint else load_checkpoint(args["arg_num"], False, device)
    )
    output_dir = Path(options.output_dir) if options.output_dir else (
        PROJECT_ROOT / "generated" / args["dataset"] / args["method"] / f"args{args['arg_num']}"
    )
    manifest, count = generate(args, checkpoint, output_dir, options.count, options.t_distance, device)
    print(f"Generated {count} images. Manifest: {manifest}")


if __name__ == "__main__":
    main()
