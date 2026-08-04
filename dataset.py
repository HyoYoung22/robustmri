"""Unified data loading for BraTS modalities, NFBS, and knee MRI."""

from glob import glob
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


DATASET_PROFILES = {
    "t1": {"root": "brats2020", "modality": "t1", "axis": 2, "slice_index": 96, "crop": 235},
    "t2": {"root": "brats2020", "modality": "t2", "axis": 2, "slice_index": 96, "crop": 235},
    "flair": {"root": "brats2020", "modality": "flair", "axis": 2, "slice_index": 96, "crop": 235},
    "nfbs": {"root": "nfbs", "modality": "t1", "axis": 1, "slice_index": 80, "crop": 235},
    "knee": {"root": "knee", "modality": "knee", "axis": 0, "slice_index": None, "crop": None},
}


def cycle(iterable):
    while True:
        yield from iterable


def _data_root(args):
    return Path(args.get("data_root") or Path(__file__).parent / "datasets")


def init_datasets(_root_dir, args):
    dataset_name = args.get("dataset", "t1").lower()
    if dataset_name in ("mri", "brain"):
        dataset_name = "t1"
    if dataset_name not in DATASET_PROFILES:
        raise ValueError(f"Unsupported dataset '{dataset_name}'")

    normalize = args.get("method", "rgdm") != "vst"
    common = {
        "dataset_name": dataset_name,
        "img_size": args["img_size"],
        "random_slice": args["random_slice"],
        "normalize": normalize,
        "data_root": _data_root(args),
    }
    return MRIDataset(split="Train", **common), MRIDataset(split="Test", **common)


def init_dataset_loader(mri_dataset, args, shuffle=True):
    return cycle(torch.utils.data.DataLoader(
        mri_dataset,
        batch_size=args["Batch_Size"],
        shuffle=shuffle,
        num_workers=0,
        drop_last=True,
    ))


class MRIDataset(Dataset):
    """Dataset with dataset-specific file discovery and shared image transforms."""

    def __init__(
        self, dataset_name, split, img_size, random_slice=False,
        normalize=True, data_root=None, transform=None,
    ):
        self.dataset_name = dataset_name
        self.profile = DATASET_PROFILES[dataset_name]
        self.split = split
        self.random_slice = random_slice
        self.data_root = Path(data_root or Path(__file__).parent / "datasets")
        self.root = self.data_root / self.profile["root"] / split
        self.subject_ids = sorted(path.name for path in self.root.iterdir() if path.is_dir())

        transform_steps = [transforms.ToPILImage()]
        if self.profile["crop"]:
            transform_steps.append(transforms.CenterCrop(self.profile["crop"]))
        transform_steps.extend([
            transforms.Resize(img_size, transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ])
        if normalize:
            transform_steps.append(transforms.Normalize((0.5,), (0.5,)))
        self.transform = transform or transforms.Compose(transform_steps)

        # Keep compatibility caches separate from caches produced by the
        # previous unified normalization formula.
        self.cache_dir = self.data_root / "cache" / "original" / dataset_name / split.lower()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self):
        return len(self.subject_ids)

    def _source_path(self, subject_id):
        subject_dir = self.root / subject_id
        if self.dataset_name in ("t1", "t2", "flair"):
            candidates = glob(str(subject_dir / f"*_{self.profile['modality']}.nii*"))
        elif self.dataset_name == "nfbs":
            prepared = subject_dir / f"{subject_id}.npy"
            if prepared.exists():
                return prepared
            candidates = glob(str(subject_dir / f"sub-{subject_id}_ses-NFB3_T1w.nii*"))
        else:
            candidates = glob(str(subject_dir / "*.npy"))
        if not candidates:
            raise FileNotFoundError(f"No source image found for {self.dataset_name}/{self.split}/{subject_id}")
        return Path(candidates[0])

    @staticmethod
    def _normalize_volume(image):
        mean = float(np.mean(image))
        std = float(np.std(image))
        lower, upper = mean - std, mean + 2 * std
        image = np.clip(image, lower, upper)
        # Preserve the formula used by the original dataset modules. It did
        # not subtract the lower clipping bound before division.
        return (image / max(upper - lower, 1e-8)).astype(np.float32)

    def _load_volume(self, subject_id):
        cache_path = self.cache_dir / f"{subject_id}.npy"
        if cache_path.exists():
            try:
                return np.load(cache_path)
            except (OSError, ValueError, EOFError):
                cache_path.unlink()

        source_path = self._source_path(subject_id)
        if source_path.suffix == ".npy":
            try:
                image = np.load(source_path)
            except (OSError, ValueError, EOFError) as exc:
                raise ValueError(f"Invalid NPY source: {source_path}") from exc
            # Original NFBS and knee loaders used prepared NPY values as-is.
            return image
        else:
            image = nib.load(str(source_path)).get_fdata()
        image = self._normalize_volume(image)
        np.save(cache_path, image)
        return image

    def __getitem__(self, index):
        subject_id = self.subject_ids[index]
        volume = self._load_volume(subject_id)
        axis = self.profile["axis"]
        configured_index = self.profile["slice_index"]
        slice_index = volume.shape[axis] // 2 if configured_index is None else configured_index
        if slice_index >= volume.shape[axis]:
            raise IndexError(
                f"Slice {slice_index} is outside axis {axis} (size {volume.shape[axis]}) "
                f"for {self.dataset_name}/{self.split}/{subject_id}"
            )
        image = np.take(volume, slice_index, axis=axis).astype(np.float32)
        image = self.transform(image)
        return {
            "image": image,
            "filenames": subject_id,
            "slice_index": slice_index,
            "dataset": self.dataset_name,
        }
