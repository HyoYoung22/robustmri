#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Final full-train script for 2D binary segmentation on .npy images/masks.

Purpose
- After K-fold evaluation, train the final model on the full train pool.
- No validation split here.
- Recommended pipeline:
    1) Run K-fold to estimate performance / choose settings
    2) Run this script on full train pool
    3) Evaluate on fixed test set

Supported backbones
  - unet_native
  - smp_unet
  - smp_unetpp
  - smp_deeplabv3p
  - segformer_b0
  - segformer_b2
  - mask2former_exp
"""

import os
import json
import random
import argparse
from pathlib import Path
from glob import glob
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# =========================
# Defaults
# =========================
PROJECT_ROOT = Path(__file__).parent
TRAIN_POOL_IMAGE_DIR = PROJECT_ROOT / 'segment_gaussian' / 'train120' / 'image'
TRAIN_POOL_MASK_DIR  = PROJECT_ROOT / 'segment_gaussian' / 'train120' / 'mask'

SAVE_DIR        = PROJECT_ROOT / 'checkpoints_compare_final'
EXP_NAME        = 'seg_compare_final'

BATCH_SIZE      = 8
NUM_EPOCHS      = 150
LEARNING_RATE   = 1e-4
NUM_WORKERS     = max(1, (os.cpu_count() or 2) // 2)
PIN_MEMORY      = True
USE_ALBU        = True
POS_WEIGHT      = None
SEED            = 42


# =========================================================
# Utils
# =========================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def is_binary_mask(arr: np.ndarray) -> bool:
    unique_vals = np.unique(arr)
    return np.all(np.isin(unique_vals, [0, 1]))

def get_sample_id(path: str) -> str:
    """
    Examples:
      99_fake.npy      -> 99
      99_true_mask.npy -> 99
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    return stem.split('_')[0]

def pair_image_mask_paths(image_dir, mask_dir):
    image_paths = sorted(glob(os.path.join(image_dir, '*.npy')))
    mask_paths  = sorted(glob(os.path.join(mask_dir, '*.npy')))

    assert len(image_paths) > 0, f"No images found in {image_dir}"
    assert len(mask_paths) > 0, f"No masks found in {mask_dir}"

    def get_sample_id(path: str) -> str:
        """
        Make matching ids from filenames.

        Examples
        100_true.npy       -> 100_true
        100_fake.npy       -> 100_fake
        100_true_mask.npy  -> 100_true
        100_fake_mask.npy  -> 100_fake
        """
        stem = os.path.splitext(os.path.basename(path))[0]

        if stem.endswith('_mask'):
            stem = stem[:-5]  # remove trailing "_mask"

        return stem
    img_map = {}
    for p in image_paths:
        sid = get_sample_id(p)
        if sid in img_map:
            raise ValueError(f"Duplicate image sample id detected: {sid} -> {p}")
        img_map[sid] = p

    msk_map = {}
    for p in mask_paths:
        sid = get_sample_id(p)
        if sid in msk_map:
            raise ValueError(f"Duplicate mask sample id detected: {sid} -> {p}")
        msk_map[sid] = p

    common_ids = sorted(set(img_map.keys()) & set(msk_map.keys()))

    assert len(common_ids) > 0, (
        "No matched sample ids between image and mask. "
        "Expected patterns like '123_fake.npy' and '123_true_mask.npy'."
    )

    image_only = sorted(set(img_map.keys()) - set(msk_map.keys()), key=lambda x: int(x))
    mask_only  = sorted(set(msk_map.keys()) - set(img_map.keys()), key=lambda x: int(x))

    print(f"✅ Paired {len(common_ids)} image-mask samples.")
    print(f"   image only: {len(image_only)}")
    print(f"   mask only : {len(mask_only)}")

    if len(image_only) > 0:
        print(f"   first image-only ids: {image_only[:10]}")
    if len(mask_only) > 0:
        print(f"   first mask-only ids : {mask_only[:10]}")

    paired_imgs = [img_map[sid] for sid in common_ids]
    paired_msks = [msk_map[sid] for sid in common_ids]

    return paired_imgs, paired_msks


# =========================================================
# Native U-Net
# =========================================================
class UNetNative(nn.Module):
    def __init__(self, in_ch=1, base_ch=64, out_ch=1, norm='batch'):
        super().__init__()
        Norm2d = nn.BatchNorm2d if norm == 'batch' else nn.InstanceNorm2d

        def CBR(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1, bias=False),
                Norm2d(o),
                nn.ReLU(inplace=True)
            )

        self.enc1 = nn.Sequential(CBR(in_ch, base_ch), CBR(base_ch, base_ch))
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(CBR(base_ch, base_ch * 2), CBR(base_ch * 2, base_ch * 2))
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = nn.Sequential(CBR(base_ch * 2, base_ch * 4), CBR(base_ch * 4, base_ch * 4))
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = nn.Sequential(CBR(base_ch * 4, base_ch * 8), CBR(base_ch * 8, base_ch * 8))

        self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 2, 2)
        self.dec3 = nn.Sequential(CBR(base_ch * 8, base_ch * 4), CBR(base_ch * 4, base_ch * 4))

        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, 2)
        self.dec2 = nn.Sequential(CBR(base_ch * 4, base_ch * 2), CBR(base_ch * 2, base_ch * 2))

        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, 2)
        self.dec1 = nn.Sequential(CBR(base_ch * 2, base_ch), CBR(base_ch, base_ch))

        self.final = nn.Conv2d(base_ch, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b  = self.bottleneck(self.pool3(e3))

        d3 = self.dec3(torch.cat([self.up3(b), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.final(d1)


# =========================================================
# Dataset
# =========================================================
class NPYDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None, replicate_to_3ch=False):
        assert len(image_paths) == len(mask_paths), "image/mask counts differ"
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.replicate_to_3ch = replicate_to_3ch

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = np.load(self.image_paths[idx]).astype(np.float32)
        msk = np.load(self.mask_paths[idx]).astype(np.float32)

        if img.max() > 1.0:
            img = img / 255.0

        if not is_binary_mask(msk):
            msk = (msk > 0.5).astype(np.float32)

        if img.ndim == 2:
            img_hwc = img[..., None]
        elif img.ndim == 3 and img.shape[0] == 1:
            img_hwc = np.transpose(img, (1, 2, 0))
        else:
            raise ValueError(f"Expected 2D grayscale or (1,H,W) image, got {img.shape}")

        if msk.ndim == 2:
            msk_hwc = msk[..., None]
        elif msk.ndim == 3 and msk.shape[0] == 1:
            msk_hwc = np.transpose(msk, (1, 2, 0))
        else:
            raise ValueError(f"Expected 2D mask or (1,H,W), got {msk.shape}")

        if self.transform is not None:
            try:
                aug = self.transform(image=img_hwc, mask=msk_hwc)
                img_hwc, msk_hwc = aug["image"], aug["mask"]
            except Exception as e:
                print(f"[WARN] Albumentations failed; using raw sample: {e}")

        img_ch = np.transpose(img_hwc, (2, 0, 1)).astype(np.float32)
        if self.replicate_to_3ch:
            img_ch = np.repeat(img_ch, 3, axis=0)

        msk_ch = np.transpose(msk_hwc, (2, 0, 1)).astype(np.float32)

        return torch.from_numpy(img_ch), torch.from_numpy(msk_ch)


def build_transforms(use_albu=True):
    if not use_albu:
        return None
    try:
        import albumentations as A
        tf = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.10,
                rotate_limit=15,
                border_mode=0,
                p=0.5
            ),
        ])
        return tf
    except Exception as e:
        print(f"[WARN] Albumentations unavailable: {e}")
        return None


# =========================================================
# Model factory
# =========================================================
class ModelWrapper(nn.Module):
    def __init__(self, kind: str, in_channels: int = 1, num_classes: int = 1, encoder_name: str = 'resnet34'):
        super().__init__()
        kind = kind.lower()
        self.kind = kind

        if kind == 'unet_native':
            self.model = UNetNative(in_ch=in_channels, base_ch=64, out_ch=num_classes, norm='batch')

        elif kind in ['smp_unet', 'smp_unetpp', 'smp_deeplabv3p']:
            try:
                import segmentation_models_pytorch as smp
            except Exception as e:
                raise RuntimeError(
                    "segmentation_models_pytorch is required. "
                    "Install: pip install segmentation-models-pytorch timm"
                ) from e

            if kind == 'smp_unet':
                self.model = smp.Unet(encoder_name=encoder_name, in_channels=in_channels, classes=num_classes)
            elif kind == 'smp_unetpp':
                self.model = smp.UnetPlusPlus(encoder_name=encoder_name, in_channels=in_channels, classes=num_classes)
            else:
                self.model = smp.DeepLabV3Plus(encoder_name=encoder_name, in_channels=in_channels, classes=num_classes)

        elif kind.startswith('segformer'):
            try:
                from transformers import SegformerForSemanticSegmentation, SegformerConfig
            except Exception as e:
                raise RuntimeError(
                    "transformers is required for SegFormer. Install: pip install transformers"
                ) from e

            model_id = 'nvidia/segformer-b2-finetuned-ade-512-512' if 'b2' in kind \
                       else 'nvidia/segformer-b0-finetuned-ade-512-512'

            config = SegformerConfig.from_pretrained(
                model_id,
                num_labels=num_classes,
                ignore_mismatched_sizes=True
            )
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                model_id,
                config=config,
                ignore_mismatched_sizes=True
            )

        elif kind == 'mask2former_exp':
            try:
                from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerConfig
            except Exception as e:
                raise RuntimeError(
                    "transformers with Mask2Former required. Install: pip install transformers"
                ) from e

            model_id = 'facebook/mask2former-swin-small-coco-instance'
            config = Mask2FormerConfig.from_pretrained(model_id)
            config.num_labels = num_classes
            self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
                model_id,
                config=config,
                ignore_mismatched_sizes=True
            )
        else:
            raise ValueError(f"Unknown model kind: {kind}")

    def forward(self, x):
        if self.kind in ['unet_native', 'smp_unet', 'smp_unetpp', 'smp_deeplabv3p']:
            return self.model(x)

        if self.kind.startswith('segformer'):
            out = self.model(pixel_values=x)
            logits = out.logits
            if logits.shape[-2:] != x.shape[-2:]:
                logits = F.interpolate(logits, size=x.shape[-2:], mode='bilinear', align_corners=False)
            return logits

        if self.kind == 'mask2former_exp':
            out = self.model(pixel_values=x)
            class_logits = out.class_queries_logits
            if class_logits.shape[-1] >= 2:
                fg_score = class_logits[..., 0]
            else:
                fg_score = class_logits.squeeze(-1)

            masks = out.pred_masks
            fg_score = fg_score[..., None, None]
            logit_map = (masks * fg_score).sum(dim=1, keepdim=True)

            if logit_map.shape[-2:] != x.shape[-2:]:
                logit_map = F.interpolate(logit_map, size=x.shape[-2:], mode='bilinear', align_corners=False)
            return logit_map

        raise RuntimeError("Unsupported model path")


# =========================================================
# Train
# =========================================================
def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    total = 0.0

    for imgs, masks in loader:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(imgs)
                loss = criterion(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(imgs)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

        total += loss.item()

    return total / max(1, len(loader))


# =========================================================
# Runner
# =========================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_pool_img_dir', default=TRAIN_POOL_IMAGE_DIR)
    parser.add_argument('--train_pool_msk_dir', default=TRAIN_POOL_MASK_DIR)

    parser.add_argument('--save_dir', default=SAVE_DIR)
    parser.add_argument('--exp', default=EXP_NAME)

    parser.add_argument('--model', default='smp_deeplabv3p',
                        choices=['unet_native', 'smp_unet', 'smp_unetpp', 'smp_deeplabv3p',
                                 'segformer_b0', 'segformer_b2', 'mask2former_exp'])
    parser.add_argument('--encoder', default='resnet50')

    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    parser.add_argument('--bs', type=int, default=BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    parser.add_argument('--pos_weight', type=float, default=POS_WEIGHT if POS_WEIGHT is not None else None)
    parser.add_argument('--seed', type=int, default=SEED)

    parser.add_argument('--no_amp', action='store_true')
    parser.add_argument('--no_albu', action='store_true')

    args = parser.parse_args()

    set_seed(args.seed)
    ensure_dir(args.save_dir)

    train_images, train_masks = pair_image_mask_paths(args.train_pool_img_dir, args.train_pool_msk_dir)
    print(f"✅ Loaded {len(train_images)} total training samples.")

    needs_rgb = args.model.startswith('segformer') or args.model.startswith('mask2former')
    train_tf = build_transforms(use_albu=(not args.no_albu))

    train_ds = NPYDataset(
        train_images,
        train_masks,
        transform=train_tf,
        replicate_to_3ch=needs_rgb
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.bs,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=(len(train_ds) >= args.bs),
        persistent_workers=(NUM_WORKERS > 0)
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    in_ch = 1 if not needs_rgb else 3
    model = ModelWrapper(
        kind=args.model,
        in_channels=in_ch,
        num_classes=1,
        encoder_name=args.encoder
    ).to(device)

    if args.pos_weight is not None:
        pw = torch.tensor([args.pos_weight], device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    else:
        criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    use_amp = (not args.no_amp) and (device.type == 'cuda')
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    run_dir = os.path.join(
        args.save_dir,
        f"{args.exp}_{args.model}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    ensure_dir(run_dir)

    last_ckpt = os.path.join(run_dir, "last.pt")
    best_train_ckpt = os.path.join(run_dir, "best_train_loss.pt")
    history_path = os.path.join(run_dir, "train_history.json")

    best_train_loss = float("inf")
    best_epoch = -1
    history = []

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)

        history.append({
            "epoch": epoch + 1,
            "train_loss": float(train_loss)
        })

        torch.save({
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "train_loss": train_loss,
            "config": {
                "lr": args.lr,
                "batch_size": args.bs,
                "amp": use_amp,
                "pos_weight": args.pos_weight,
                "model": args.model,
                "encoder": args.encoder,
                "seed": args.seed,
            }
        }, last_ckpt)

        if train_loss < best_train_loss:
            best_train_loss = train_loss
            best_epoch = epoch + 1
            torch.save({
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict() if scaler is not None else None,
                "train_loss": train_loss,
                "config": {
                    "lr": args.lr,
                    "batch_size": args.bs,
                    "amp": use_amp,
                    "pos_weight": args.pos_weight,
                    "model": args.model,
                    "encoder": args.encoder,
                    "seed": args.seed,
                }
            }, best_train_ckpt)

        print(
            f"Epoch {epoch+1:03d}/{args.epochs} | "
            f"TrainLoss {train_loss:.4f} | "
            f"BestTrainLoss {best_train_loss:.4f} (E{best_epoch})"
        )

    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print("=" * 80)
    print("✅ Final full-data training finished.")
    print(f"Last checkpoint      : {last_ckpt}")
    print(f"Best train-loss ckpt : {best_train_ckpt}")
    print(f"History saved to     : {history_path}")


if __name__ == "__main__":
    main()
