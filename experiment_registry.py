"""Resolve dataset and method implementations from experiment arguments.

This module deliberately keeps the existing dataset and diffusion modules intact.
It only centralises the combinations that used to be encoded in separate scripts.
"""

from importlib import import_module


DATASET_ALIASES = {
    "mri": "t1",
    "brain": "t1",
    "t1": "t1",
    "t2": "t2",
    "flair": "flair",
    "nfbs": "nfbs",
    "knee": "knee",
}
METHOD_ALIASES = {
    "rgdm": "rgdm",
    "vst": "vst",
}


def normalise_experiment_args(args):
    """Validate and normalise the dataset/method values in an args mapping."""
    dataset_name = str(args.get("dataset", "mri")).lower()
    method_name = str(args.get("method", "rgdm")).lower()

    try:
        dataset_name = DATASET_ALIASES[dataset_name]
    except KeyError as exc:
        supported = ", ".join(sorted(DATASET_ALIASES))
        raise ValueError(f"Unsupported dataset '{dataset_name}'. Choose one of: {supported}") from exc

    try:
        method_name = METHOD_ALIASES[method_name]
    except KeyError as exc:
        supported = ", ".join(sorted(METHOD_ALIASES))
        raise ValueError(f"Unsupported method '{method_name}'. Choose one of: {supported}") from exc

    args["dataset"] = dataset_name
    args["method"] = method_name
    return dataset_name, method_name


def load_experiment_components(args):
    """Return the dataset and common diffusion implementations."""
    dataset_name, method_name = normalise_experiment_args(args)

    diffusion_module = "GaussianDiffusion"

    return (
        import_module("dataset"),
        import_module(diffusion_module),
    )
