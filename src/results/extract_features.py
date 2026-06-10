"""Extract frozen encoder features from trained single-branch checkpoint(s).

Loads every seed checkpoint in a multi-seed run directory, runs every data split
through the (frozen) encoder in **eval mode**, and caches the feature vectors to
disk so a downstream fusion module can be trained on precomputed features instead
of re-running the encoder each epoch.

The model decides which branch is exported: ``SingleBranchModel`` (HR) writes
``fmow_rgb`` features, ``SingleBranchLRModel`` (LR) writes ``landsat`` features.
For each seed run, features are written under the parent directory of
``preprocessed_dir``:

    <parent>/FMoW_LandSat_<run-name>_Features/run<i>/fmow_features/fmow_rgb/rgb_img_<file_idx>.pt   (HR)
    <parent>/FMoW_LandSat_<run-name>_Features/run<i>/fmow_features/landsat/image_<file_idx>.pt       (LR)

Files are keyed by the global ``file_idx`` (``full_idxs``) and named to mirror the
existing preprocessed-image convention, so the same loader logic can read them.
Splits are disjoint over ``file_idx``, so all splits share the branch's flat dir;
seeds are kept apart by the ``run<i>`` level because each seed is a different
trained model and therefore yields different features for the same image.

Everything runs in eval mode on purpose: cached features must be deterministic
(BatchNorm running stats, dropout disabled). Any regularization for the
downstream fusion module belongs inside that module, not baked into the cache.

Usage:
    uv run --env-file .env src/results/extract_features.py <run_dir> [--batch-size N] [--overwrite]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from lightning import seed_everything
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataset.fmow_multiscale_dataset import resolve_preprocessed_dir
from models.components.fusion_model import SingleBranchLRModel, SingleBranchModel
from results.utils import find_best_checkpoints, load_hydra_config
from train.run_experiment import _parse_spatial_cfg, make_model
from train.utils import make_multiscale_dataset


SEED = 111
BATCH_SIZE = 64
DATA_LOADER_NUM_WORKERS = 4
NUM_RERUNS = 3


class _IndexedSubset(torch.utils.data.Dataset):
    """Wraps a WILDS subset so each item carries its global ``file_idx``.

    Returns only ``(file_idx, x)`` — labels/metadata are irrelevant for feature
    extraction and the branch encoders consume the ``x`` dict alone.
    """

    def __init__(self, subset, full_idxs):
        self.subset = subset
        self.full_idxs = full_idxs

    def __len__(self) -> int:
        return len(self.subset)

    def file_idx_at(self, i: int) -> int:
        """Resolve the global ``file_idx`` for subset position ``i`` without
        loading the (heavy) image tensors."""
        dataset_idx = int(self.subset.indices[i])
        return int(self.full_idxs[dataset_idx])

    def __getitem__(self, i: int):
        x, _, _ = self.subset[i]
        return self.file_idx_at(i), x


def _collate(batch):
    file_idxs, xs = zip(*batch)
    keys = xs[0].keys()
    x_batch = {k: torch.stack([x[k] for x in xs], dim=0) for k in keys}
    return list(file_idxs), x_batch


def find_seed_checkpoints(path: Path) -> list[tuple[int, Path]]:
    """Resolve each seed's best checkpoint via find_best_checkpoints.

    A run directory holds ``checkpoints/run0``, ``checkpoints/run1``, ... each with
    a best ``late-fusion-*.ckpt`` (falling back to ``last.ckpt``). Each checkpoint is
    paired with the seed index parsed from its ``run<i>`` parent dir. Runs train
    exactly ``NUM_RERUNS`` seeds, so a directory resolving to a different count is
    treated as an incomplete/wrong run and raises. Returns
    ``[(seed_idx, ckpt_path), ...]`` sorted by seed.
    """
    if path.is_file():
        raise ValueError(f"Expected a run directory, got a file: {path}")
    if not path.is_dir():
        raise FileNotFoundError(f"Run directory not found: {path}")
    if not (path / "checkpoints").is_dir():
        raise ValueError(f"Expected a run directory with checkpoints/, got {path}")

    checkpoints = find_best_checkpoints(path)
    if len(checkpoints) != NUM_RERUNS:
        raise ValueError(
            f"Expected {NUM_RERUNS} seed checkpoints under {path / 'checkpoints'}, "
            f"found {len(checkpoints)} ({sorted(c.parent.name for c in checkpoints)}). "
            "Point at a completed multi-seed run directory."
        )
    return sorted((int(c.parent.name[3:]), c) for c in checkpoints)


def build_dataset(cfg):
    """Build the eval-mode (augment=False) multiscale dataset, matching the
    spatial-feature configuration the checkpoint was trained with."""
    sc = _parse_spatial_cfg(cfg)
    return make_multiscale_dataset(
        fmow_dir=cfg.data.fmow_dir,
        landsat_dir=cfg.data.landsat_dir,
        source=cfg.data.get("source", "preprocessed"),
        preprocessed_dir=cfg.data.preprocessed_dir,
        augment=False,
        image_norm=cfg.data.image_norm,
        lr_crop_km=cfg.data.get("lr_crop_km", None),
        spatial_coord_grid=sc["needs_coord_grid"],
        spatial_overlap_mask=sc["needs_overlap_mask"],
        overlap_mask_type=sc["overlap_mask_type"],
        lr_extension_factor=cfg.data.lr_extension_factor,
    )


def resolve_output_base(cfg) -> Path:
    """The ``FMoW_LandSat_<run-name>_Features`` dir; per-seed run<i> is added later.

    Sits next to the (host-resolved) preprocessed dir, mirroring resolve_feature_dir.
    """
    preprocessed_dir = resolve_preprocessed_dir(cfg.data.preprocessed_dir)
    if preprocessed_dir is None:
        raise ValueError(
            "cfg.data.preprocessed_dir is None; cannot derive the output location."
        )
    run_name = cfg.run_name.replace("train_", "")
    return Path(preprocessed_dir).parent / f"FMoW_LandSat_{run_name.title()}_Features"


def resolve_branch(model):
    """Map a single-branch model to its feature export spec.

    Returns ``(subdir, filename, feature_fn)`` where ``subdir`` is the
    ``fmow_features/`` subdirectory, ``filename(file_idx)`` builds the per-sample
    filename (mirroring the preprocessed-image convention), and ``feature_fn(x)``
    runs the encoder on the right input. Raises for unsupported model types.
    """
    if isinstance(model, SingleBranchLRModel):
        n = model.landsat_channels
        return (
            "landsat",
            lambda file_idx: f"image_{file_idx}.pt",
            lambda x: model.encoder(x["landsat"][:, :n, :, :]),
        )
    if isinstance(model, SingleBranchModel):
        return (
            "fmow_rgb",
            lambda file_idx: f"rgb_img_{file_idx}.pt",
            lambda x: model.encoder(x["rgb"]),
        )
    raise ValueError(
        "Feature extraction supports single-branch models "
        f"(SingleBranchModel / SingleBranchLRModel); got {type(model).__name__}."
    )


@torch.no_grad()
def extract_split(branch, dataset, split, out_dir, args, device) -> None:
    _, filename, feature_fn = branch
    subset = dataset.get_subset(split, frac=1.0)
    if len(subset) == 0:
        print(f"  [{split}] empty, skipping.")
        return

    indexed = _IndexedSubset(subset, dataset.full_idxs)

    if args.overwrite:
        pending = list(range(len(indexed)))
    else:
        pending = [
            i
            for i in range(len(indexed))
            if not (out_dir / filename(indexed.file_idx_at(i))).exists()
        ]

    skipped = len(indexed) - len(pending)
    if not pending:
        print(f"  [{split}] {len(indexed)} samples already cached, skipping.")
        return

    loader = DataLoader(
        Subset(indexed, pending),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=_collate,
    )

    desc = f"  [{split}] {len(pending)} to extract" + (
        f" ({skipped} cached)" if skipped else ""
    )
    for file_idxs, x in tqdm(loader, desc=desc):
        x = {k: v.to(device, non_blocking=True) for k, v in x.items()}
        feats = feature_fn(x).cpu()
        for j, file_idx in enumerate(file_idxs):
            # .clone() detaches the slice from the batch storage; without it
            # torch.save would serialize the whole batch into every file.
            torch.save(feats[j].clone(), out_dir / filename(file_idx))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache frozen single-branch (HR or LR) encoder features from checkpoint(s) (eval mode)."
    )
    parser.add_argument("checkpoint_path", type=str, help="Path to a multi-seed run directory (containing checkpoints/run*)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Inference batch size")
    parser.add_argument("--num-workers", type=int, default=DATA_LOADER_NUM_WORKERS, help="DataLoader workers")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    parser.add_argument("--overwrite", action="store_true", help="Recompute and overwrite existing feature files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # frac=1.0 with shuffle=False makes the sample set seed-independent, so unlike
    # eval_reproduce.py we don't re-seed per run; one seed_everything suffices.
    seed_everything(args.seed, workers=True)

    run_dir = Path(args.checkpoint_path)
    seed_checkpoints = find_seed_checkpoints(run_dir)
    cfg = load_hydra_config(run_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Architecture is identical across seeds: build once, reload weights per seed.
    module = make_model(cfg)
    branch = resolve_branch(module.model)
    module.to(device)

    dataset, preprocessed_dir = build_dataset(cfg)
    out_base = resolve_output_base(cfg, preprocessed_dir)
    splits = list(dataset._split_dict.keys())

    subdir = branch[0]
    print(f"=== Output base ===\n  {out_base}")
    print(f"=== Branch ===\n  {subdir}")
    print(f"=== Seeds ===\n  {[i for i, _ in seed_checkpoints]}")
    print(f"=== Splits ===\n  {splits}")

    for seed_idx, ckpt_path in seed_checkpoints:
        print(f"\n--- seed {seed_idx} ({ckpt_path}) ---")
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        module.load_state_dict(checkpoint["state_dict"])
        module.eval()

        out_dir = out_base / f"run{seed_idx}" / "fmow_features" / subdir
        out_dir.mkdir(parents=True, exist_ok=True)

        for split in splits:
            extract_split(branch, dataset, split, out_dir, args, device)

    print("\nDone.")


if __name__ == "__main__":
    main()
