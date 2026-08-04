# RGDM

PyTorch research code for MRI training and generation with the Rician–Gaussian Diffusion Model (RGDM), comparison with the independent VST baseline, generated-image quality assessment, and segmentation experiments.

## Publication

This work was published in the journal *Neural Networks* in 2026:

> Hyoyoung Jang and Sangmin Lee, “Diffusion model with Rician–Gaussian priors for robust MR image synthesis,” *Neural Networks*, vol. 203, article 109163, 2026. DOI: [10.1016/j.neunet.2026.109163](https://www.sciencedirect.com/science/article/pii/S0893608026006246)

## Features

- Supports T1, T2, FLAIR, NFBS, and Knee MRI datasets
- Selects RGDM or the VST baseline through a shared execution pipeline
- Trains diffusion models and resumes from checkpoints
- Generates images from trained models
- Evaluates generated images using MSE, PSNR, and SSIM
- Trains, cross-validates, and evaluates segmentation models using generated images

## Project Structure

```text
RGDM/
├── datasets/
│   ├── brats2020/              # Shared source data for T1, T2, and FLAIR
│   │   ├── Train/
│   │   └── Test/
│   ├── nfbs/
│   │   ├── Train/
│   │   └── Test/
│   ├── knee/
│   │   ├── Train/
│   │   └── Test/
│   ├── cache/                  # Modality-specific preprocessing cache
│   └── legacy/                 # Preserved copies of legacy data
├── model/                      # Diffusion checkpoints
├── test_args/                  # Experiment JSON configurations
├── GaussianDiffusion.py        # Shared RGDM/VST diffusion implementation
├── UNet.py
├── dataset.py                  # Unified loader for all five datasets
├── experiment_registry.py
├── train_diffusion.py
├── generate_images.py
├── evaluate_generation.py
├── train_segmentation.py
├── cross_validate_segmentation.py
└── evaluate_segmentation.py
```

All default paths are resolved relative to the `RGDM` directory. User-specific absolute paths such as `/home/...` are not used.

## Installation

Python 3.9 or newer and a CUDA-enabled PyTorch environment are recommended.

```bash
pip install torch torchvision numpy matplotlib nibabel opencv-python \
  scikit-image scikit-learn scipy pandas imageio pytorch-msssim
```

Additional packages may be required depending on the segmentation model.

```bash
pip install albumentations segmentation-models-pytorch timm transformers
```

## Datasets

The following `dataset` values are supported.

| Value | Source directory | Slice axis | Description |
|---|---|---:|---|
| `t1` | `datasets/brats2020` | 2 | BraTS2020 T1 |
| `t2` | `datasets/brats2020` | 2 | BraTS2020 T2 |
| `flair` | `datasets/brats2020` | 2 | BraTS2020 FLAIR |
| `nfbs` | `datasets/nfbs` | 1 | NFBS T1 |
| `knee` | `datasets/knee` | 0 | Knee MRI |

T1, T2, and FLAIR share the same BraTS2020 spatial structure and preprocessing pipeline. The modality is selected by filename.

```text
*_t1.nii
*_t2.nii
*_flair.nii
```

Preprocessing caches are separated by modality to prevent data from being mixed.

```text
datasets/cache/t1/
datasets/cache/t2/
datasets/cache/flair/
datasets/cache/nfbs/
datasets/cache/knee/
```

## Experiment Configuration

Experiments are configured with `test_args/argsNUM.json` files.

```json
{
  "dataset": "t1",
  "method": "rgdm",
  "img_size": [256, 256],
  "Batch_Size": 1,
  "EPOCHS": 3000,
  "T": 1000,
  "base_channels": 128,
  "beta_schedule": "linear",
  "loss-type": "hybrid",
  "loss_weight": "none",
  "lr": 0.0001,
  "random_slice": false,
  "sample_distance": 1000,
  "noise_fn": "rician"
}
```

### Supported Values

- `dataset`: `t1`, `t2`, `flair`, `nfbs`, `knee`
- `method`: `rgdm`, `vst`
- `noise_fn`: `rician`, `gauss`
- `beta_schedule`: `linear`, `cosine`
- `loss-type`: `l1`, `l2`, `hybrid`

If an existing configuration does not specify `method`, `rgdm` is used by default.

## 1. Train a Diffusion Model

All three argument formats below are supported.

```bash
python train_diffusion.py 38
python train_diffusion.py args38
python train_diffusion.py args38.json
```

Resume from the most recent checkpoint:

```bash
python train_diffusion.py RESUME_RECENT args38
```

Continue training from the final model:

```bash
python train_diffusion.py RESUME_FINAL args38
```

Output locations:

```text
model/diff-params-ARGS=38/params-final.pt
model/diff-params-ARGS=38/checkpoint/
diffusion-training-images/ARGS=38/
diffusion-videos/ARGS=38/
```

After training completes successfully, intermediate checkpoints are removed and `params-final.pt` is retained.

## 2. Generate Images

```bash
python generate_images.py args38 --count 100 --t-distance 500
```

Configuration values can also be overridden from the command line.

```bash
python generate_images.py args38 \
  --dataset flair \
  --method vst \
  --checkpoint model/diff-params-ARGS=38/params-final.pt \
  --count 100 \
  --t-distance 500
```

Default output structure:

```text
generated/<dataset>/<method>/args<NUM>/
├── input/
│   ├── *.npy
│   └── *.png
├── generated/
│   ├── *.npy
│   └── *.png
└── manifest.csv
```

`manifest.csv` records each sample ID, dataset, method, source path, and generated-image path.

## 3. Evaluate Generated-Image Quality

```bash
python evaluate_generation.py \
  generated/t1/rgdm/args38/manifest.csv
```

Specify a custom output path:

```bash
python evaluate_generation.py MANIFEST.csv --output quality_metrics.csv
```

Evaluation metrics:

- MSE
- PSNR
- SSIM

By default, results are saved as `quality_metrics.csv` in the same directory as the manifest.

## 4. Segmentation Experiments

### Train a Final Model on the Full Training Set

```bash
python train_segmentation.py \
  --train_pool_img_dir segment_rgdm/train/image \
  --train_pool_msk_dir segment_rgdm/train/mask \
  --model smp_deeplabv3p \
  --encoder resnet50 \
  --epochs 150
```

Supported models:

- `unet_native`
- `smp_unet`
- `smp_unetpp`
- `smp_deeplabv3p`
- `segformer_b0`
- `segformer_b2`
- `mask2former_exp`

### K-Fold Cross-Validation

```bash
python cross_validate_segmentation.py \
  --train_pool_img_dir segment_rgdm/train/image \
  --train_pool_msk_dir segment_rgdm/train/mask \
  --n_splits 5 \
  --use_group_kfold
```

### Final Evaluation

```bash
python evaluate_segmentation.py \
  --val_img_dir segment/test/image \
  --val_msk_dir segment/test/mask \
  --ckpt checkpoints_compare/best.pt \
  --model unet_native \
  --out_dir eval_out
```

## End-to-End Workflow

```text
1. Create test_args/argsNUM.json
2. Run train_diffusion.py
3. Run generate_images.py
4. Run evaluate_generation.py
5. Run train_segmentation.py or cross_validate_segmentation.py
6. Run evaluate_segmentation.py
```

## Notes

- Preserve patient-level separation between the training and test sets.
- Prevent generated images derived from test-set patients from entering the training split.
- VST is an independent comparison method, not a preprocessing option for RGDM.
- VST dataset inputs remain in `[0, 1]` and are normalized separately in diffusion space after the VST transformation.
- Some legacy Knee and cached `.npy` files may be empty or truncated. The unified loader removes damaged central cache entries and attempts to regenerate them from the source data. A sample cannot be loaded when no valid Knee source exists.
- `datasets/legacy` preserves previous data copies and is not used by the current training code.

## Command Help

```bash
python generate_images.py --help
python evaluate_generation.py --help
python train_segmentation.py --help
python cross_validate_segmentation.py --help
python evaluate_segmentation.py --help
```
