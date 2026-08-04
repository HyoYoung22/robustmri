#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation script for the unified 2D binary segmentation project.
- Computes per-image and aggregate metrics: Dice, IoU (mIoU in binary), Precision, Recall, F1, Pixel Accuracy
- **Adds** boundary metrics: HD95 (95th percentile Hausdorff Distance), ASD (Average Surface Distance)
- Reports mean and standard deviation (표준편차) for each metric
- Saves a CSV of per-image metrics
- Saves triplet visualizations: [Input, GT mask, Pred mask]

This file is self-contained (copies Dataset/ModelWrapper structure). Put it next to your training code.

Example:
python seg_eval.py \
  --val_img_dir ./segment/test/image \
  --val_msk_dir ./segment/test/mask \
  --ckpt ./checkpoints_compare/seg_compare_unet_native_20250101-120000/best.pt \
  --model unet_native --encoder resnet34 --thr 0.5 --spacing 1.0,1.0 \
  --out_dir ./eval_out --save_n 32

Note: HD95/ASD require SciPy: `pip install scipy`
"""

import os, sys, argparse, math, random
from pathlib import Path
from glob import glob
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import csv
import imageio.v2 as imageio
# =========================
# Dataset & small utils
# =========================

def is_binary_mask(arr: np.ndarray) -> bool:
    u = np.unique(arr)
    return np.all(np.isin(u, [0, 1]))

class NPYDataset(Dataset):
    """Single-channel .npy images & masks.
    replicate_to_3ch=True -> (1,H,W) -> (3,H,W) for backbones that require RGB.
    """
    def __init__(self, image_paths, mask_paths, replicate_to_3ch=False):
        assert len(image_paths) == len(mask_paths), "image/mask counts differ"
        self.image_paths = image_paths
        self.mask_paths  = mask_paths
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

        # to CHW
        if img.ndim == 2:
            img_ch = img[None, ...]
        elif img.ndim == 3 and img.shape[0] == 1:
            img_ch = img
        else:
            raise ValueError("Expected 2D or (1,H,W) input")

        if self.replicate_to_3ch:
            img_ch = np.repeat(img_ch, 3, axis=0)

        if msk.ndim == 2:
            msk_ch = msk[None, ...]
        elif msk.ndim == 3 and msk.shape[0] == 1:
            msk_ch = msk
        else:
            raise ValueError("Expected 2D or (1,H,W) mask")

        return torch.from_numpy(img_ch), torch.from_numpy(msk_ch), os.path.basename(self.image_paths[idx])

# =========================
# Simple native U-Net (same as train)
# =========================
class UNetNative(nn.Module):
    def __init__(self, in_ch=1, base_ch=64, out_ch=1, norm='batch'):
        super().__init__()
        Norm2d = nn.BatchNorm2d if norm == 'batch' else nn.InstanceNorm2d
        def CBR(i,o):
            return nn.Sequential(
                nn.Conv2d(i,o,3,padding=1,bias=False),
                Norm2d(o),
                nn.ReLU(inplace=True)
            )
        self.enc1 = nn.Sequential(CBR(in_ch, base_ch), CBR(base_ch, base_ch))
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(CBR(base_ch, base_ch*2), CBR(base_ch*2, base_ch*2))
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = nn.Sequential(CBR(base_ch*2, base_ch*4), CBR(base_ch*4, base_ch*4))
        self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = nn.Sequential(CBR(base_ch*4, base_ch*8), CBR(base_ch*8, base_ch*8))
        self.up3 = nn.ConvTranspose2d(base_ch*8, base_ch*4, 2, 2)
        self.dec3 = nn.Sequential(CBR(base_ch*8, base_ch*4), CBR(base_ch*4, base_ch*4))
        self.up2 = nn.ConvTranspose2d(base_ch*4, base_ch*2, 2, 2)
        self.dec2 = nn.Sequential(CBR(base_ch*4, base_ch*2), CBR(base_ch*2, base_ch*2))
        self.up1 = nn.ConvTranspose2d(base_ch*2, base_ch, 2, 2)
        self.dec1 = nn.Sequential(CBR(base_ch*2, base_ch), CBR(base_ch, base_ch))
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

# =========================
# Model wrapper (same interface as train)
# =========================
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
                raise RuntimeError("Install segmentation-models-pytorch + timm for SMP backbones") from e
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
                raise RuntimeError("Install transformers for SegFormer backbones") from e
            model_id = 'nvidia/segformer-b2-finetuned-ade-512-512' if 'b2' in kind else 'nvidia/segformer-b0-finetuned-ade-512-512'
            config = SegformerConfig.from_pretrained(model_id, num_labels=num_classes, ignore_mismatched_sizes=True)
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                model_id, config=config, ignore_mismatched_sizes=True
            )

        elif kind == 'mask2former_exp':
            try:
                from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerConfig
            except Exception as e:
                raise RuntimeError("Install transformers>=4.29 for Mask2Former") from e
            model_id = 'facebook/mask2former-swin-small-coco-instance'
            config = Mask2FormerConfig.from_pretrained(model_id)
            config.num_labels = num_classes
            self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
                model_id, config=config, ignore_mismatched_sizes=True
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
            if class_logits.shape[-1] >= 2:  # pick foreground logit heuristically
                fg_score = class_logits[..., 0]
            else:
                fg_score = class_logits.squeeze(-1)
            masks = out.pred_masks
            fg_score = fg_score[..., None, None]
            logit_map = (masks * fg_score).sum(dim=1, keepdim=True)
            if logit_map.shape[-2:] != x.shape[-2:]:
                logit_map = F.interpolate(logit_map, size=x.shape[-2:], mode='bilinear', align_corners=False)
            return logit_map
        raise RuntimeError("Unsupported model kind")

# =========================
# Metrics (pixel + boundary)
# =========================
@torch.no_grad()
def binarize(sigmoid_logits: torch.Tensor, thr: float) -> torch.Tensor:
    return (torch.sigmoid(sigmoid_logits) > thr).float()

@torch.no_grad()
def dice_coeff(pred: torch.Tensor, target: torch.Tensor, eps=1e-6) -> torch.Tensor:
    inter = (pred * target).sum(dim=(1,2,3))
    union = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))
    return (2*inter + eps) / (union + eps)

@torch.no_grad()
def iou_score(pred: torch.Tensor, target: torch.Tensor, eps=1e-6) -> torch.Tensor:
    inter = (pred * target).sum(dim=(1,2,3))
    union = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3)) - inter
    return (inter + eps) / (union + eps)

@torch.no_grad()
def precision_recall_f1(pred: torch.Tensor, target: torch.Tensor, eps=1e-6):
    tp = (pred*target).sum(dim=(1,2,3))
    fp = (pred*(1-target)).sum(dim=(1,2,3))
    fn = ((1-pred)*target).sum(dim=(1,2,3))
    prec = (tp + eps) / (tp + fp + eps)
    rec  = (tp + eps) / (tp + fn + eps)
    f1   = (2*prec*rec) / (prec + rec + eps)
    return prec, rec, f1

@torch.no_grad()
def pixel_accuracy(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # pred, target: (B,1,H,W) with {0,1}
    return (pred == target).float().mean(dim=(1,2,3))

# ---- Boundary metrics helpers (HD95/ASD) ----

def _require_scipy():
    try:
        from scipy import ndimage as ndi  # noqa: F401
    except Exception:
        raise RuntimeError("HD95/ASD require scipy. Install with `pip install scipy`.")

def _binary_boundary(mask: np.ndarray) -> np.ndarray:
    from scipy import ndimage as ndi
    mask = mask.astype(bool)
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    eroded = ndi.binary_erosion(mask, structure=np.ones((3,3), dtype=bool), iterations=1, border_value=0)
    boundary = mask ^ eroded
    return boundary

def _surface_distances(gt: np.ndarray, pr: np.ndarray, spacing: Tuple[float,float]) -> np.ndarray:
    """Symmetric surface distances between binary masks (2D)."""
    from scipy import ndimage as ndi
    gt = gt.astype(bool)
    pr = pr.astype(bool)

    # Empty-handling: return image diagonal if exactly one is empty
    if not gt.any() and not pr.any():
        return np.array([0.0], dtype=np.float64)
    if gt.any() and not pr.any():
        h, w = gt.shape
        diag = math.hypot(spacing[0]*(h-1), spacing[1]*(w-1))
        return np.array([diag], dtype=np.float64)
    if pr.any() and not gt.any():
        h, w = pr.shape
        diag = math.hypot(spacing[0]*(h-1), spacing[1]*(w-1))
        return np.array([diag], dtype=np.float64)

    gt_bd = _binary_boundary(gt)
    pr_bd = _binary_boundary(pr)
    if not gt_bd.any() and not pr_bd.any():
        return np.array([0.0], dtype=np.float64)

    dt_gt = ndi.distance_transform_edt(~gt_bd, sampling=spacing)
    dt_pr = ndi.distance_transform_edt(~pr_bd, sampling=spacing)

    d_pr_to_gt = dt_gt[pr_bd]
    d_gt_to_pr = dt_pr[gt_bd]

    if d_pr_to_gt.size == 0 and d_gt_to_pr.size == 0:
        return np.array([0.0], dtype=np.float64)
    return np.concatenate([d_pr_to_gt, d_gt_to_pr]).astype(np.float64)

def hd95_asd_batch(pred_bin: torch.Tensor, target_bin: torch.Tensor, spacing: Tuple[float,float]):
    _require_scipy()
    p = pred_bin.detach().cpu().numpy().astype(np.uint8)
    t = target_bin.detach().cpu().numpy().astype(np.uint8)
    B = p.shape[0]
    hd, asd = [], []
    for i in range(B):
        d = _surface_distances(t[i,0], p[i,0], spacing)
        hd.append(float(np.percentile(d, 95)) if d.size else 0.0)
        asd.append(float(d.mean()) if d.size else 0.0)
    return np.array(hd, dtype=np.float64), np.array(asd, dtype=np.float64)

# =========================
# Visualization
# =========================

def save_individual_images(out_dir, name, img_chw, gt_chw, pred_chw):
    base = name.replace(".npy", "")

    input_dir = os.path.join(out_dir, "input")
    gt_dir = os.path.join(out_dir, "gt_mask")
    pred_dir = os.path.join(out_dir, "pred_mask")

    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)

    img = img_chw.detach().cpu().numpy()
    gt = gt_chw.detach().cpu().numpy()[0]
    pr = pred_chw.detach().cpu().numpy()[0]

    # first channel 사용
    show = img[0] if img.shape[0] >= 1 else img

    # 0~255 uint8 변환
    show_u8 = (np.clip(show, 0, 1) * 255).astype(np.uint8)
    gt_u8 = (np.clip(gt, 0, 1) * 255).astype(np.uint8)
    pr_u8 = (np.clip(pr, 0, 1) * 255).astype(np.uint8)

    imageio.imwrite(os.path.join(input_dir, f"{base}.png"), show_u8)
    imageio.imwrite(os.path.join(gt_dir, f"{base}.png"), gt_u8)
    imageio.imwrite(os.path.join(pred_dir, f"{base}.png"), pr_u8)
# =========================
# Main
# =========================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--val_img_dir', required=True)
    ap.add_argument('--val_msk_dir', required=True)
    ap.add_argument('--ckpt', required=True, help='Path to best.pt (or directory containing best.pt)')
    ap.add_argument('--model', default='unet_native',
                   choices=['unet_native','smp_unet','smp_unetpp','smp_deeplabv3p','segformer_b0','segformer_b2','mask2former_exp'])
    ap.add_argument('--encoder', default='resnet34')
    ap.add_argument('--thr', type=float, default=0.5)
    ap.add_argument('--spacing', type=str, default='1.0,1.0', help='Pixel spacing as "row,col" (e.g., 0.5,0.5)')
    ap.add_argument('--bs', type=int, default=1)
    ap.add_argument('--num_workers', type=int, default=max(1, (os.cpu_count() or 2)//2))
    ap.add_argument('--out_dir', default=Path(__file__).parent / 'eval_out')
    ap.add_argument('--save_n', type=int, default=32, help='How many triplets to save (0 to disable)')
    args = ap.parse_args()

    # parse spacing
    try:
        spacing = tuple(float(x) for x in args.spacing.split(','))
        assert len(spacing) == 2
    except Exception:
        raise ValueError('Invalid --spacing. Use e.g. "1.0,1.0"')

    os.makedirs(args.out_dir, exist_ok=True)

    # Resolve checkpoint file
    ckpt_path = args.ckpt
    if os.path.isdir(ckpt_path):
        best = os.path.join(ckpt_path, 'best.pt')
        last = os.path.join(ckpt_path, 'last.pt')
        if os.path.isfile(best):
            ckpt_path = best
        elif os.path.isfile(last):
            ckpt_path = last
        else:
            raise FileNotFoundError('No best.pt/last.pt found in directory: ' + args.ckpt)

    # Build file lists
    val_images = sorted(glob(os.path.join(args.val_img_dir, '*.npy')))
    val_masks  = sorted(glob(os.path.join(args.val_msk_dir,  '*.npy')))
    assert len(val_images) > 0 and len(val_images) == len(val_masks), 'Validation set mismatch or empty'

    needs_rgb = args.model.startswith('segformer') or args.model.startswith('mask2former')
    ds = NPYDataset(val_images, val_masks, replicate_to_3ch=needs_rgb)
    dl = DataLoader(ds, batch_size=args.bs, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    in_ch = 3 if needs_rgb else 1
    model = ModelWrapper(kind=args.model, in_channels=in_ch, num_classes=1, encoder_name=args.encoder).to(device)

    print(f'Loading checkpoint: {ckpt_path}')
    state = torch.load(ckpt_path, map_location=device)
    sd = state['model'] if isinstance(state, dict) and 'model' in state else state
    model.load_state_dict(sd, strict=False)
    model.eval()

    all_dice, all_iou, all_prec, all_rec, all_f1 = [], [], [], [], []
    all_hd95, all_asd = [], []
    all_acc = []

    save_count = 0
    csv_path = os.path.join(args.out_dir, 'per_image_metrics.csv')
    with open(csv_path, 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['filename','dice','iou','precision','recall','f1','hd95','asd','acc'])

        with torch.no_grad():
            for imgs, masks, names in dl:
                imgs  = imgs.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                logits = model(imgs)
                preds = binarize(logits, args.thr)

                # pixel metrics (tensor)
                d = dice_coeff(preds, masks)
                i = iou_score(preds, masks)
                p, r, f1 = precision_recall_f1(preds, masks)
                acc = pixel_accuracy(preds, masks)

                # boundary metrics (numpy, needs scipy)
                try:
                    hd_b, asd_b = hd95_asd_batch(preds, masks, spacing)
                except RuntimeError as e:
                    print('[WARN] Boundary metrics skipped:', e)
                    hd_b = np.zeros((preds.shape[0],), dtype=np.float64)
                    asd_b = np.zeros((preds.shape[0],), dtype=np.float64)

                # accumulate per-sample (batch may be >1)
                d_np  = d.detach().cpu().numpy()
                i_np  = i.detach().cpu().numpy()
                p_np  = p.detach().cpu().numpy()
                r_np  = r.detach().cpu().numpy()
                f1_np = f1.detach().cpu().numpy()
                acc_np = acc.detach().cpu().numpy()

                for bi in range(len(names)):
                    wr.writerow([
                        names[bi], float(d_np[bi]), float(i_np[bi]), float(p_np[bi]), float(r_np[bi]), float(f1_np[bi]),
                        float(hd_b[bi]), float(asd_b[bi]), float(acc_np[bi])
                    ])
                    all_dice.append(float(d_np[bi]))
                    all_iou.append(float(i_np[bi]))
                    all_prec.append(float(p_np[bi]))
                    all_rec.append(float(r_np[bi]))
                    all_f1.append(float(f1_np[bi]))
                    all_hd95.append(float(hd_b[bi]))
                    all_asd.append(float(asd_b[bi]))
                    all_acc.append(float(acc_np[bi]))

                    # ✅ 개별 이미지별 결과 콘솔 출력 추가
                    print(f"[{names[bi]}]")
                    print(f"  Dice={d_np[bi]:.4f}, IoU={i_np[bi]:.4f}, Prec={p_np[bi]:.4f}, Rec={r_np[bi]:.4f}, "
                        f"F1={f1_np[bi]:.4f}, Acc={acc_np[bi]:.4f}, HD95={hd_b[bi]:.3f}, ASD={asd_b[bi]:.3f}")
                    print("-" * 80)

                    # visualization (only for a subset)
                    save_individual_images(
                        args.out_dir,
                        names[bi],
                        imgs[bi].cpu(),
                        masks[bi].cpu(),
                        preds[bi].cpu()
                    )
                    save_count += 1

    # Aggregate stats: mean & std (population std)
    def mean_std(x):
        x = np.array(x, dtype=np.float64)
        if x.size == 0:
            return 0.0, 0.0
        return float(x.mean()), float(x.std(ddof=0))

    md, sd = mean_std(all_dice)
    mi, si = mean_std(all_iou)
    mp, sp = mean_std(all_prec)
    mr, sr = mean_std(all_rec)
    mf, sf = mean_std(all_f1)
    mh, sh = mean_std(all_hd95)
    ma, sa = mean_std(all_asd)
    macc, sacc = mean_std(all_acc)

    # Save summary txt
    summary_path = os.path.join(args.out_dir, 'summary.txt')
    with open(summary_path, 'w') as g:
        g.write('== Segmentation Evaluation Summary ==\n')
        g.write(f'Num samples: {len(all_dice)}\n')
        g.write(f'Threshold: {args.thr}\n')
        g.write(f'Spacing: row={spacing[0]}, col={spacing[1]}\n\n')
        g.write(f'Dice : mean={md:.4f}, std={sd:.4f}\n')
        g.write(f'IoU  : mean={mi:.4f}, std={si:.4f}\n')
        g.write(f'Prec.: mean={mp:.4f}, std={sp:.4f}\n')
        g.write(f'Rec. : mean={mr:.4f}, std={sr:.4f}\n')
        g.write(f'F1   : mean={mf:.4f}, std={sf:.4f}\n')
        g.write(f'HD95 : mean={mh:.4f}, std={sh:.4f}\n')
        g.write(f'ASD  : mean={ma:.4f}, std={sa:.4f}\n')
        g.write(f'Acc. : mean={macc:.4f}, std={sacc:.4f}\n')

    print('== Summary ==')
    print(f'Dice : mean={md:.4f}, std={sd:.4f}')
    print(f'IoU  : mean={mi:.4f}, std={si:.4f}')
    print(f'Prec.: mean={mp:.4f}, std={sp:.4f}')
    print(f'Rec. : mean={mr:.4f}, std={sr:.4f}')
    print(f'F1   : mean={mf:.4f}, std={sf:.4f}')
    print(f'HD95 : mean={mh:.4f}, std={sh:.4f}')
    print(f'ASD  : mean={ma:.4f}, std={sa:.4f}')
    print(f'Acc. : mean={macc:.4f}, std={sacc:.4f}')
    print(f"Saved per-image metrics CSV -> {csv_path}")
    print(f"Saved summary               -> {summary_path}")
    if args.save_n > 0:
        print(f"Saved {min(args.save_n, len(all_dice))} visualizations to {args.out_dir}")

if __name__ == '__main__':
    main()
