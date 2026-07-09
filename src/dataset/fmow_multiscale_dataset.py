"""FMoW/Landsat multi-scale PyTorch Dataset.

Defines `FMoWMultiScaleDataset`, a WILDS-compatible dataset that pairs each
FMoW-WILDS high-resolution (HR) RGB satellite image with a broader-scale
low-resolution (LR) Landsat GeoTIFF covering the same location, for this
thesis's multi-scale fusion models. The dataset can serve three mutually
exclusive input representations per branch (`source="raw"|"preprocessed"|
"features"`): raw images read from disk and transformed on the fly,
pre-transformed tensors cached by the preprocessing pipeline, or pooled
encoder features cached by `extract_features.py`. It can also emit optional
spatial encodings (per-pixel coordinate grids and an HR/LR overlap mask) for
spatially-aware model variants, and supports a geographic
leave-one-continent-out split (`leave_asia_out`) and a spatial-extent-ablation
crop of the LR branch (`lr_crop_km`).

Helper functions:
    `_host_data_root`: Resolve the machine-specific base data directory.
    `resolve_preprocessed_dir`: Build the host-specific preprocessed-data path.
    `resolve_feature_dir`: Build the host-specific cached-feature path for a
        given run.
    `collate_multiscale`: `DataLoader` collate function matching this
        dataset's `__getitem__` output, stacking per-branch tensors (or
        passing through `None` for branches unused in feature mode).
"""
import platform
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision.transforms import v2 as transforms
import rasterio
from wilds.datasets.wilds_dataset import WILDSDataset


DEFAULT_PREPROCESSED_DIR = "FMoW_LandSat"

def _host_data_root() -> str:
    """Resolve the base data directory for the current machine.

    Different hosts (SLURM/compute nodes) mount the FMoW/Landsat datasets at
    different paths; this maps `platform.node()` to that host's data root.

    Returns:
        str: Absolute path to the host's data root directory.

    Raises:
        ValueError: If the current host name is not one of the known hosts.
    """
    node = platform.node()
    if node in {"gaia4", "gaia5", "gaia6", "gaia7"}:
        return "/data/henicke"
    if node in {"nyx"}:
        return "/home/nyx_data1/henicke"
    if node in {"gaia1"}:
        return "/users/henicke"
    if node in {"kallisto", "io"}:
        return "/home/datasets4/FMoW_LandSat"
    raise ValueError(f"Unknown host {node}, cannot resolve data path.")


def resolve_preprocessed_dir(preprocessed_dir: str | None = DEFAULT_PREPROCESSED_DIR) -> str | None:
    """Resolve a preprocessed-dataset directory name to a host-specific absolute path.

    Args:
        preprocessed_dir (str | None): Base directory name under the host's
            data root (e.g. `"FMoW_LandSat"`), or `None` to skip resolution.
            Defaults to `DEFAULT_PREPROCESSED_DIR`.

    Returns:
        str | None: `"<host_data_root>/<preprocessed_dir>"`, or `None` if
        `preprocessed_dir` is `None`.
    """
    if preprocessed_dir is None:
        return None
    return f"{_host_data_root()}/{preprocessed_dir}"


def resolve_feature_dir(run_name: str, run_idx: int, branch_subdir: str) -> Path:
    """Build the host-specific path to cached encoder features for one branch.

    Mirrors `resolve_preprocessed_dir`'s host map:
    ``<data-root>/FMoW_LandSat_<run_name.title()>_Features/run<run_idx>/fmow_features/<branch_subdir>``,
    where ``branch_subdir`` is ``fmow_rgb`` (HR) or ``landsat`` (LR). Features
    are written by `extract_features.py`.

    Args:
        run_name (str): Name of the feature-extraction run (title-cased into
            the directory name).
        run_idx (int): Index of the rerun whose features to load (0/1/2 ->
            run0/run1/run2).
        branch_subdir (str): Branch subdirectory to point at: `"fmow_rgb"`
            for the HR branch or `"landsat"` for the LR branch.

    Returns:
        Path: Absolute path to the cached-feature directory for this
        run/branch.
    """
    return (
        Path(_host_data_root())
        / f"FMoW_LandSat_{run_name.title()}_Features"
        / f"run{run_idx}"
        / "fmow_features"
        / branch_subdir
    )


class FMoWMultiScaleDataset(WILDSDataset):
    """WILDS-compatible dataset pairing each FMoW-WILDS sample with a co-located Landsat image.

    Extends the FMoW-WILDS dataset with a second, broader-scale input branch:
    for every FMoW high-resolution (HR) RGB image (224x224, 3 channels) it
    also serves a Landsat GeoTIFF (6 bands) covering a wider ground footprint
    around the same location, at either native or 224x224 resolution
    depending on configuration. Reuses the base WILDS dataset's splits,
    labels and metadata array (extended here with `lat`/`lon`/`img_span_km`
    columns from `extended_metadata_csv`), and inherits from `WILDSDataset`
    (e.g. `get_subset`) for compatibility with the WILDS evaluation tooling.

    Each sample can be served in one of three mutually exclusive
    representations (`source`): raw images transformed on the fly, tensors
    pre-transformed and cached to disk, or pooled encoder features cached by
    `extract_features.py`. Optional spatial encodings (per-pixel coordinate
    grids and an HR/LR overlap mask) and a leave-one-continent-out split
    variant are also supported; see `__init__` for the corresponding
    parameters.

    Attributes:
        _dataset_name (str): WILDS dataset identifier (`"fmow_multiscale"`).
        _IMG_SIZE (int): Side length in pixels that RGB and (by default)
            Landsat tensors are resized to (224).
        _LANDSAT_FULLRES_SIZE (int): Side length in pixels of the stored
            full-resolution Landsat tensors used by `lr_crop_km` (497).
        _SOURCES (tuple[str, ...]): Valid values for the `source` constructor
            argument: `("raw", "preprocessed", "features")`.
    """

    _dataset_name = "fmow_multiscale"

    _IMG_SIZE = 224

    # Stored full-res Landsat image size (see save_fullres_landsat.py): 497x497.
    _LANDSAT_FULLRES_SIZE = 497

    # Mutually-exclusive input modes; ``source`` selects exactly one (see __init__).
    _SOURCES = ("raw", "preprocessed", "features")

    def __init__(
        self,
        fmow_dir="data",
        landsat_dir="data",
        source="raw",
        preprocessed_dir=None,
        extended_metadata_csv="rgb_metadata_extended.csv",
        split_scheme="official",
        transform_rgb=None,
        transform_landsat=None,
        augment=False,
        hflip_prob=0.5,
        vflip_prob=0.5,
        image_norm="fmow-statistics",
        scale_to_img_size=True,
        lr_crop_km=None,
        spatial_coord_grid=False,
        spatial_overlap_mask=False,
        overlap_mask_type="binary",
        lr_extension_factor=3.0,
        hr_feature_run_name=None,
        lr_feature_run_name=None,
        feature_run_idx=None,
        leave_asia_out=False,
    ):
        """Initialize the dataset: load base WILDS FMoW metadata and configure input/spatial options.

        Args:
            fmow_dir (str): Root directory of the FMoW-WILDS dataset, passed
                to `wilds.get_dataset`; also used to locate raw HR images
                (``<fmow_dir>/fmow_v<version>/images``) when ``source="raw"``.
                Defaults to ``"data"``.
            landsat_dir (str): Root directory for Landsat data; used to
                locate raw LR images (``<landsat_dir>/fmow_landsat/images``)
                when ``source="raw"`` and to load ``extended_metadata_csv``
                regardless of ``source``. Defaults to ``"data"``.
            source (str): Which input mode to serve (mutually exclusive):
                - ``"raw"``: read raw images from ``fmow_dir`` (HR) and
                  ``landsat_dir`` (LR); transforms applied on the fly.
                - ``"preprocessed"``: read pre-transformed tensors from the single
                  ``preprocessed_dir`` (``fmow_preprocessed/{fmow_rgb,landsat}``).
                - ``"features"``: read cached encoder features (written by
                  extract_features.py); see ``hr_feature_run_name``/``lr_feature_run_name``.
                Only the dirs for the active mode are read; the others are ignored.
                Defaults to ``"raw"``.
            preprocessed_dir (str | None): Base dir name for ``source="preprocessed"``
                (required there); host-resolved internally via
                `resolve_preprocessed_dir`, so pass the config name (e.g.
                ``"FMoW_LandSat"``), not an absolute path. Defaults to `None`.
            extended_metadata_csv (str): Filename, relative to
                ``<landsat_dir>/fmow_landsat/``, of the CSV providing per-sample
                ``lat``/``lon``/``img_span_km`` (appended to the base WILDS
                metadata array/fields). Rows with ``split == "seq"`` are
                excluded. Defaults to ``"rgb_metadata_extended.csv"``.
            split_scheme (str): 'official' (time_after_2016) or other WILDS
                split schemes, forwarded to `wilds.get_dataset`. Defaults to
                ``"official"``.
            transform_rgb (Callable | None): Transform applied to raw HR
                images (``source="raw"``). Defaults to
                `get_default_transform_rgb()` if `None`.
            transform_landsat (Callable | None): Transform applied to raw LR
                images (``source="raw"``). Defaults to
                `get_default_transform_landsat()` if `None`.
            augment (bool): If True, apply a shared random horizontal/vertical
                flip to the HR/LR images (and spatial tensors) in
                `__getitem__`, skipped when ``source="features"``. Defaults
                to `False`.
            hflip_prob (float): Probability of a horizontal flip when
                `augment` is True. Defaults to 0.5.
            vflip_prob (float): Probability of a vertical flip when
                `augment` is True. Defaults to 0.5.
            image_norm (str): Normalization statistics scheme used by
                `get_default_transform_rgb`/`get_default_transform_landsat`:
                ``"fmow-statistics"`` (dataset-specific mean/std), ``"const"``
                (fixed 0.5/0.5, RGB only), or any other value (ImageNet stats
                for RGB, theoretical DN-to-reflectance stats for Landsat).
                Defaults to ``"fmow-statistics"``.
            scale_to_img_size (bool): If True (default), Landsat images are resized to
                ``_IMG_SIZE`` (224). If False, they are kept at their native
                resolution (e.g. 497x497) — still normalized, just not downscaled.
                Only valid with ``source="raw"`` (the preprocessed/feature paths
                serve already-sized tensors); raises otherwise. Used by the
                full-res preprocessing step for the spatial-extent ablation.
            lr_crop_km (float | None): If set, center-crop the stored full-res Landsat tensor to
                this spatial extent (km) and resize the crop back to ``_IMG_SIZE``.
                Only valid with ``source="preprocessed"`` pointing at the full-res
                set (normalized 497x497). Must lie in ``(0, full_span]`` where the
                full footprint is inferred from metadata (max HR span x
                ``lr_extension_factor``), so ``lr_extension_factor`` is required.
                This is the knob for the spatial-extent ablation: a smaller value
                keeps a tighter centered patch (then up/downsampled to 224). When
                spatial features are on, it also defines ``lr_span_km`` so the coord
                grid / overlap mask track the cropped footprint. Defaults to `None`.
            spatial_coord_grid (bool): If True, precompute and emit per-pixel
                coordinate grid tensors (``coord_grid_lr``/``coord_grid_hr``, in
                ``[-1, 1]``) from `__getitem__`. Not compatible with
                ``source="features"``. Defaults to `False`.
            spatial_overlap_mask (bool): If True, emit an ``overlap_mask``
                tensor marking where the HR footprint falls inside the LR
                image, shaped by `overlap_mask_type`. Not compatible with
                ``source="features"``. Defaults to `False`.
            overlap_mask_type (str): ``"binary"`` (hard box mask, default) or
                ``"gaussian"`` (soft Gaussian falloff), used when
                `spatial_overlap_mask` is True.
            lr_extension_factor (float): Multiplier applied to the
                metadata-inferred max HR image span to obtain the full LR
                (Landsat) footprint in km, used both to validate/size
                `lr_crop_km` and, when no crop is set, as `lr_span_km` for the
                spatial encodings. Defaults to 3.0.
            hr_feature_run_name (str | None): For ``source="features"``: run-name of the cached HR
                features (the ``<run>`` in ``FMoW_LandSat_<hr_feature_run_name>_Features``). When
                set, ``x["rgb"]`` is the precomputed HR feature vector. Defaults to `None`.
            lr_feature_run_name (str | None): As ``hr_feature_run_name`` for the LR/landsat branch
                (``x["landsat"]``). Defaults to `None`.
            feature_run_idx (int | None): For ``source="features"``: index of the rerun to load
                (0/1/2 -> run0/run1/run2). Required there. Leaving one of the two
                run-names ``None`` yields a ``None`` tensor for that branch (e.g.
                retraining a single-branch classifier on the other modality).
                Defaults to `None`.
            leave_asia_out (bool): If True, turn the WILDS split scheme into a geographic
                leave-one-continent-out split while keeping the split names intact:
                Asia samples are dropped from ``train``/``id_val``/``val`` and
                non-Asia samples are dropped from ``id_test``/``test`` (set to the
                ``-1`` unused code). The model then trains and is model-selected on
                non-Asia and is evaluated exclusively on the unseen Asia continent.
                Defaults to `False`.

        Raises:
            ValueError: If ``source`` is not one of `_SOURCES`; if
                ``scale_to_img_size=False`` is combined with a ``source``
                other than ``"raw"``; if ``source="preprocessed"`` and
                ``preprocessed_dir`` is `None`; if ``source="features"`` and
                ``feature_run_idx`` is `None`/negative, or both
                ``hr_feature_run_name``/``lr_feature_run_name`` are `None`;
                if ``lr_crop_km`` is set with ``source`` other than
                ``"preprocessed"``, or lies outside ``(0, full_span_km]``; or
                if a spatial encoding (`spatial_coord_grid`/
                `spatial_overlap_mask`) is requested with ``source="features"``.
            FileNotFoundError: If the resolved HR or LR cached-feature
                directory does not exist when ``source="features"``.
        """
        from wilds import get_dataset

        # Initialize base FMoW dataset to get splits and metadata
        self.base_dataset = get_dataset(
            dataset="fmow", root_dir=fmow_dir, download=False, split_scheme=split_scheme
        )

        self.root_fmow = Path(fmow_dir) / f"fmow_v{self.base_dataset.version}"
        self.root_landsat = Path(landsat_dir) / "fmow_landsat"
        self.image_norm = image_norm

        # Input mode. Exactly one of raw / preprocessed / features is active; each
        # reads only its own dirs (see get_rgb_input / get_landsat_input). The base
        # roots above are still needed for metadata regardless of mode.
        if source not in self._SOURCES:
            raise ValueError(f"source must be one of {self._SOURCES}, got {source!r}")
        self.source = source

        # Full-res toggle for the Landsat branch; only the raw path resizes images,
        # so disabling the downscale is meaningful only there.
        if not scale_to_img_size and self.source != "raw":
            raise ValueError("scale_to_img_size=False is only valid with self.source='raw'.")
        self.scale_to_img_size = scale_to_img_size

        # raw: per-branch image dirs.
        self.fmow_images = self.root_fmow / "images"
        self.landsat_images = self.root_landsat / "images"
        # preprocessed: one base dir serving both branches.
        self.fmow_images_preprocessed = None
        self.landsat_images_preprocessed = None
        # features: per-branch cached-feature dirs (None side -> None tensor).
        self.hr_features_dir = None
        self.lr_features_dir = None

        if self.source == "preprocessed":
            if preprocessed_dir is None:
                raise ValueError("preprocessed_dir is required when self.source='preprocessed'.")

            base = Path(resolve_preprocessed_dir(preprocessed_dir))
            self.fmow_images_preprocessed = base / "fmow_preprocessed" / "fmow_rgb"
            self.landsat_images_preprocessed = base / "fmow_preprocessed" / "landsat"
        elif self.source == "features":
            if feature_run_idx is None or feature_run_idx < 0:
                raise ValueError(
                    f"feature_run_idx (0/1/2) is required when self.source='features'; got {feature_run_idx!r}."
                )
            if hr_feature_run_name is None and lr_feature_run_name is None:
                raise ValueError(
                    "self.source='features' requires at least one of hr_feature_run/lr_feature_run."
                )

            if hr_feature_run_name is not None:
                self.hr_features_dir = resolve_feature_dir(hr_feature_run_name, feature_run_idx, "fmow_rgb")
                if not self.hr_features_dir.is_dir():
                    raise FileNotFoundError(f"HR feature dir not found: {self.hr_features_dir}")
            if lr_feature_run_name is not None:
                self.lr_features_dir = resolve_feature_dir(lr_feature_run_name, feature_run_idx, "landsat")
                if not self.lr_features_dir.is_dir():
                    raise FileNotFoundError(f"LR feature dir not found: {self.lr_features_dir}")

        # Inherit attributes from base dataset
        self._dataset_name = "fmow_multiscale"
        self._data_dir = str(self.root_fmow)
        self._split_scheme = self.base_dataset._split_scheme
        self._split_dict = self.base_dataset._split_dict
        self._split_names = self.base_dataset._split_names
        self._split_array = self.base_dataset._split_array
        self._y_array = self.base_dataset._y_array
        self._y_size = self.base_dataset._y_size
        self._n_classes = self.base_dataset._n_classes
        self._metadata_fields = self.base_dataset._metadata_fields
        self._metadata_array = self.base_dataset._metadata_array
        self._metadata_map = self.base_dataset._metadata_map
        self._eval_groupers = self.base_dataset._eval_groupers
        self._original_resolution = (224, 224)

        # Store full indices (accounting for sequestered images)
        self.full_idxs = self.base_dataset.full_idxs
        self.metadata = self.base_dataset.metadata

        # Extend metadata with coords and image span from the extended CSV
        csv_path = self.root_landsat / extended_metadata_csv
        ext_df = pd.read_csv(csv_path)
        ext_df = ext_df[ext_df["split"] != "seq"].reset_index(drop=True)
        extra_cols = torch.tensor(
            ext_df[["lat", "lon", "img_span_km"]].values, dtype=torch.float32
        )
        self._metadata_array = torch.cat(
            [self._metadata_array.float(), extra_cols], dim=1
        )
        self._metadata_fields = self._metadata_fields + ["lat", "lon", "img_span_km"]

        self.leave_asia_out = leave_asia_out
        if leave_asia_out:
            region_idx = self._metadata_fields.index("region")
            region_arr = self._metadata_array[:, region_idx].numpy().astype(int)
            asia_mask = region_arr == self._metadata_map["region"].index("Asia")

            # Drop Asia from the train/val splits and drop non-Asia from
            # the test splits (set to -1, the "unused" code). 
            train_val_codes = [self._split_dict[s] for s in ("train", "id_val", "val")]
            test_codes = [self._split_dict[s] for s in ("id_test", "test")]
            drop = (asia_mask & np.isin(self._split_array, train_val_codes)) | (
                ~asia_mask & np.isin(self._split_array, test_codes)
            )
            self._split_array[drop] = -1

        # Transforms
        self.transform_rgb = transform_rgb or self.get_default_transform_rgb()
        self.transform_landsat = (
            transform_landsat or self.get_default_transform_landsat()
        )

        self.augment = augment
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob

        # Infer the full Landsat footprint in km from the metadata 
        img_span_idx = self._metadata_fields.index("img_span_km")
        max_hr_span = self._metadata_array[:, img_span_idx].max().item()
        fullres_span_km = max_hr_span * lr_extension_factor

        # Load-time spatial-extent crop for the Landsat branch. Operates on the
        # full-res tensors saved by save_fullres_landsat.py (normalized 497x497),
        # so it is meaningful only on the preprocessed path. 
        self.lr_crop_km = lr_crop_km
        if self.lr_crop_km is not None:
            if self.source != "preprocessed":
                raise ValueError("lr_crop_km is only valid with self.source='preprocessed'.")
            if not 0 < self.lr_crop_km <= fullres_span_km:
                raise ValueError(
                    f"lr_crop_km must be in (0, {fullres_span_km:g}] km; got {self.lr_crop_km}."
                )
            self._crop_px = max(1, min(
                self._LANDSAT_FULLRES_SIZE,
                round(self.lr_crop_km / fullres_span_km * self._LANDSAT_FULLRES_SIZE),
            ))
            print(f"Full Landsat span inferred from metadata: {fullres_span_km:.1f} km; cropping to {self.lr_crop_km} km -> {self._crop_px}px.")

        self.spatial_coord_grid = spatial_coord_grid
        self.spatial_overlap_mask = spatial_overlap_mask
        self.overlap_mask_type = overlap_mask_type

        if self.spatial_coord_grid or self.spatial_overlap_mask:
            if self.source == "features":
                raise ValueError("Spatial encodings are not compatible with self.source='features'.")

            if self.lr_crop_km is not None:
                # The crop sets the physical LR footprint, so the coord grid and
                # overlap mask must key off the cropped extent, not the factor.
                self.lr_span_km = self.lr_crop_km
            else:
                self.lr_span_km = fullres_span_km
            S = self._IMG_SIZE
            self._lr_res = self.lr_span_km * 1000.0 / S
            self._coord_scale = (S - 1) * self._lr_res / 2.0  # half-extent for [-1, 1]
            self._coord_center = self._coord_scale  # center value in meters
            pixel = torch.arange(S, dtype=torch.float32)
            py, px = torch.meshgrid(pixel * self._lr_res, pixel * self._lr_res, indexing="ij")
            self._coord_grid_lr = (torch.stack([px, py], dim=0) - self._coord_center) / self._coord_scale  # (2, S, S), in [-1, 1]
        else:
            self.lr_span_km = fullres_span_km 


    def get_default_transform_rgb(self):
        """Build the default RGB normalization transform based on `self.image_norm`.

        Converts a PIL image to a float32 tensor scaled to ``[0, 1]`` and
        normalizes it with one of three statistics sets, selected by
        `self.image_norm`: ``"fmow-statistics"`` uses per-channel mean/std
        computed over the FMoW dataset, ``"const"`` uses a fixed 0.5/0.5
        normalization, and any other value falls back to standard ImageNet
        mean/std.

        Returns:
            torchvision.transforms.v2.Compose: Transform mapping a PIL RGB
            image to a ``(3, H, W)`` float32 normalized tensor.
        """
        if self.image_norm == "fmow-statistics":
            return transforms.Compose(
                [
                    # v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]) is equivalent to transforms.ToTensor()
                    transforms.ToImage(),
                    transforms.ToDtype(torch.float32, scale=True),
                    transforms.Normalize(
                        mean=[0.4155880808830261, 0.41815927624702454, 0.3903605341911316],
                        std=[0.24812281131744385, 0.24405813217163086, 0.2482403963804245],
                    )
                ]
            )
        elif self.image_norm == "const":
            return transforms.Compose(
                [
                    transforms.ToImage(),
                    transforms.ToDtype(torch.float32, scale=True),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ]
            )
        else:
            return transforms.Compose(
                [
                    transforms.ToImage(),
                    transforms.ToDtype(torch.float32, scale=True),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )

    def get_default_transform_landsat(self):
        """Build the default Landsat normalization transform based on `self.image_norm`.

        See https://developers.google.com/earth-engine/datasets/catalog/landsat/ for scaling info.
        GeoTIFFs already store reflectance values directly, so unlike the RGB
        transform this only normalizes (no `ToImage`/`ToDtype` conversion).
        ``"fmow-statistics"`` uses per-band mean/std computed over the full
        dataset; any other value uses a theoretical linear DN (0-65353) to
        reflectance scaling to derive a fixed mean/std applied to all 6 bands.

        Returns:
            torchvision.transforms.v2.Compose: Transform normalizing a
            ``(6, H, W)`` float32 Landsat tensor in place.
        """
        if self.image_norm == "fmow-statistics":
            return transforms.Compose(
                [
                    transforms.Normalize(
                        mean=[0.06259285658597946, 0.0880340114235878, 0.09441816806793213,
                            0.2327403724193573, 0.19073842465877533, 0.12976829707622528],
                        std=[0.039894334971904755, 0.049978554248809814, 0.0687960833311081,
                            0.092967689037323, 0.09390033036470413, 0.0819208025932312],
                    ),
                ]
            )
        else:
            # Theoretical scaling from raw DN (0–65353) to reflectance:
            landsat_scale = 2.75e-05
            landsat_offset = -0.2
            lower = 0 * landsat_scale + landsat_offset
            upper = 65353 * landsat_scale + landsat_offset
            mean = (upper + lower) / 2
            std = (upper - lower) / 2
            return transforms.Compose(
                [
                    transforms.Normalize(mean=[mean] * 6, std=[std] * 6),
                ]
            )

    def _apply_augmentation(self, rgb_img, landsat_img, spatial_tensors=None):
        """Apply a shared random horizontal/vertical flip to the HR/LR images and spatial tensors.

        Each flip is independently sampled once per call (not per tensor), so
        the RGB image, Landsat image, and any spatial tensors (e.g.
        coordinate grids, overlap mask) stay spatially consistent with each
        other.

        Args:
            rgb_img (torch.Tensor): ``(3, H, W)`` HR image tensor.
            landsat_img (torch.Tensor): ``(6, H, W)`` LR image tensor.
            spatial_tensors (dict[str, torch.Tensor] | None): Optional extra
                ``(C, H, W)`` spatial-encoding tensors (as produced by
                `_build_spatial_tensors`) to flip identically. Defaults to
                `None` (treated as ``{}``).

        Returns:
            tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]: The
            ``(rgb_img, landsat_img, spatial_tensors)`` tensors/dict after
            applying the sampled flips (`spatial_tensors` values are flipped
            in place and the same dict is returned).
        """
        if spatial_tensors is None:
            spatial_tensors = {}

        hflip = torch.rand(1).item() < self.hflip_prob
        vflip = torch.rand(1).item() < self.vflip_prob

        if hflip:
            rgb_img = torch.flip(rgb_img, dims=[2])
            landsat_img = torch.flip(landsat_img, dims=[2])
            for k in spatial_tensors:
                spatial_tensors[k] = torch.flip(spatial_tensors[k], dims=[2])

        if vflip:
            rgb_img = torch.flip(rgb_img, dims=[1])
            landsat_img = torch.flip(landsat_img, dims=[1])
            for k in spatial_tensors:
                spatial_tensors[k] = torch.flip(spatial_tensors[k], dims=[1])

        return rgb_img, landsat_img, spatial_tensors

    def _build_spatial_tensors(self, img_span_km):
        """Build optional per-sample spatial-encoding tensors for one sample.

        Called from `__getitem__` with that sample's HR image span. Returns
        an empty dict unless `self.spatial_coord_grid` or
        `self.spatial_overlap_mask` is enabled (set in `__init__`, gated to
        ``source != "features"``).

        Args:
            img_span_km (float): Ground distance (km) covered by this
                sample's HR image, used to place the HR coordinate grid
                within the shared LR footprint and to size the overlap mask.

        Returns:
            dict[str, torch.Tensor]: Zero or more of:
                "coord_grid_lr": ``(2, _IMG_SIZE, _IMG_SIZE)`` float32, the
                    dataset-wide LR pixel coordinate grid in ``[-1, 1]``
                    (`self._coord_grid_lr`, precomputed in `__init__`;
                    identical across samples). Present if
                    `self.spatial_coord_grid` is True.
                "coord_grid_hr": ``(2, _IMG_SIZE, _IMG_SIZE)`` float32, this
                    sample's HR pixel coordinate grid, in the same
                    ``[-1, 1]`` space as "coord_grid_lr" but covering only
                    the (smaller) HR footprint centered within it. Present
                    if `self.spatial_coord_grid` is True.
                "overlap_mask": ``(1, _IMG_SIZE, _IMG_SIZE)`` float32 mask
                    marking where the HR footprint falls inside the LR
                    image: a hard box (``overlap_mask_type="binary"``) or a
                    soft Gaussian falloff (``overlap_mask_type="gaussian"``),
                    sized by the ratio of the HR to LR ground span. Present
                    if `self.spatial_overlap_mask` is True.
        """
        tensors = {}
        if not (self.spatial_coord_grid or self.spatial_overlap_mask):
            return tensors

        S = self._IMG_SIZE
        lr_res = self._lr_res
        hr_res = img_span_km * 1000.0 / S
        center = S / 2.0
        offset = center * lr_res - center * hr_res

        if self.spatial_coord_grid:
            tensors["coord_grid_lr"] = self._coord_grid_lr  # (2, S, S), shared, in [-1, 1]
            pixel = torch.arange(S, dtype=torch.float32)
            py, px = torch.meshgrid(pixel * hr_res + offset, pixel * hr_res + offset, indexing="ij")
            tensors["coord_grid_hr"] = (torch.stack([px, py], dim=0) - self._coord_center) / self._coord_scale

        if self.spatial_overlap_mask:
            ratio = img_span_km / self.lr_span_km
            if self.overlap_mask_type == "gaussian":
                lin = torch.linspace(-1, 1, S)
                gy, gx = torch.meshgrid(lin, lin, indexing="ij")
                sigma = ratio / 2.0
                mask = torch.exp(-(gx ** 2 + gy ** 2) / (2 * sigma ** 2))
            else:
                lin = torch.linspace(-1, 1, S)
                gy, gx = torch.meshgrid(lin, lin, indexing="ij")
                mask = ((gx.abs() <= ratio) & (gy.abs() <= ratio)).float()
            tensors["overlap_mask"] = mask.unsqueeze(0)  # (1, S, S)

        return tensors

    def __len__(self):
        """Number of samples in the dataset.

        Returns:
            int: Number of labeled samples (``len(self._y_array)``).
        """
        return len(self._y_array)

    def __getitem__(self, idx):
        """Load one sample: HR/LR images (or features), label, and metadata.

        Maps `idx` to the underlying file id via `self.full_idxs`, loads the
        HR input via `get_rgb_input` and the LR input via
        `get_landsat_input`, then attaches per-sample metadata (lat/lon,
        region, HR image span) and optional spatial encodings from
        `_build_spatial_tensors`. When `self.augment` is enabled (and
        ``source != "features"``, since features are pooled vectors with no
        spatial extent), a shared random flip is applied to the images and
        any spatial tensors via `_apply_augmentation`.

        Args:
            idx (int): Sample index in ``[0, len(self))``, indexing
                `self._y_array` / `self._metadata_array` / `self.full_idxs`.

        Returns:
            tuple: ``(x, y, metadata)`` where:
                x (dict[str, torch.Tensor | None]): Model input dict with
                    keys:
                    "rgb": HR branch tensor from `get_rgb_input`. For
                        ``source in {"raw", "preprocessed"}``:
                        ``(3, _IMG_SIZE, _IMG_SIZE)`` float32, normalized RGB
                        image. For ``source="features"``: cached pooled
                        feature vector, or `None` if `hr_feature_run_name`
                        was not set.
                    "landsat": LR branch tensor from `get_landsat_input`. For
                        ``source in {"raw", "preprocessed"}``:
                        ``(6, _IMG_SIZE, _IMG_SIZE)`` float32 normalized
                        Landsat image (or ``(6, _LANDSAT_FULLRES_SIZE,
                        _LANDSAT_FULLRES_SIZE)`` when ``source="raw"`` and
                        `scale_to_img_size` is False). For
                        ``source="features"``: cached pooled feature vector,
                        or `None` if `lr_feature_run_name` was not set.
                    "coords": ``(2,)`` float32 tensor, ``[lat, lon]`` for
                        this sample.
                    "domain": scalar float32 tensor, the WILDS region code
                        (see `self._metadata_map["region"]`).
                    "img_span_km": scalar float32 tensor, ground span (km)
                        covered by the HR image.
                    "coord_grid_lr" / "coord_grid_hr" (optional):
                        ``(2, _IMG_SIZE, _IMG_SIZE)`` float32 tensors of
                        per-pixel (x, y) coordinates in ``[-1, 1]``, present
                        only if `self.spatial_coord_grid` is True (see
                        `_build_spatial_tensors`).
                    "overlap_mask" (optional): ``(1, _IMG_SIZE, _IMG_SIZE)``
                        float32 tensor marking where the HR footprint falls
                        inside the LR image, present only if
                        `self.spatial_overlap_mask` is True.
                y (torch.Tensor): Scalar tensor, the integer class label for
                    this sample (from the base WILDS `_y_array`).
                metadata (torch.Tensor): ``(len(self._metadata_fields),)``
                    float32 tensor, the full metadata row for this sample
                    (WILDS metadata fields followed by the appended "lat",
                    "lon", "img_span_km").
        """
        file_idx = self.full_idxs[idx]

        rgb_img = self.get_rgb_input(file_idx)
        landsat_img = self.get_landsat_input(file_idx)

        y = self._y_array[idx]
        metadata = self._metadata_array[idx]
        coords_start = self._metadata_fields.index("lat")
        coords = self._metadata_array[idx, coords_start:coords_start + 2]
        img_span_idx = self._metadata_fields.index("img_span_km")
        img_span_km = self._metadata_array[idx, img_span_idx].item()

        spatial = self._build_spatial_tensors(img_span_km)

        # Features are pooled vectors (and may be None), so spatial flips don't apply.
        if self.augment and self.source != "features":
            rgb_img, landsat_img, spatial = self._apply_augmentation(rgb_img, landsat_img, spatial)

        region_idx = self._metadata_fields.index("region")

        x = {
            "rgb": rgb_img,
            "landsat": landsat_img,
            "coords": coords,
            "domain": self._metadata_array[idx, region_idx],
            "img_span_km": self._metadata_array[idx, img_span_idx],
            **spatial,
        }

        return x, y, metadata

    def get_rgb_input(self, idx):
        """Load the HR (FMoW RGB) input for one sample under the active `source` mode.

        Args:
            idx (int): File id used to locate the sample's image/feature
                file (i.e. `self.full_idxs[dataset_idx]`, not the dataset
                index itself).

        Returns:
            torch.Tensor | None:
                - ``source="features"``: cached pooled HR feature vector
                  loaded from ``self.hr_features_dir / f"rgb_img_{idx}.pt"``,
                  or `None` if `self.hr_features_dir` is unset (no
                  `hr_feature_run_name` given).
                - ``source="preprocessed"``: pre-transformed tensor loaded
                  from ``self.fmow_images_preprocessed / f"rgb_img_{idx}.pt"``.
                - ``source="raw"``: ``(3, _IMG_SIZE, _IMG_SIZE)`` float32
                  tensor, the RGB PNG at
                  ``self.fmow_images / f"rgb_img_{idx}.png"`` after
                  `self.transform_rgb`.
        """
        if self.source == "features":
            if self.hr_features_dir is None:
                return None
            return torch.load(self.hr_features_dir / f"rgb_img_{idx}.pt", weights_only=False)

        if self.source == "preprocessed":
            return torch.load(self.fmow_images_preprocessed / f"rgb_img_{idx}.pt", weights_only=False)

        img_path = self.fmow_images / f"rgb_img_{idx}.png"
        img = Image.open(img_path).convert("RGB")

        if self.transform_rgb is not None:
            img = self.transform_rgb(img)

        return img

    def get_landsat_input(self, idx):
        """Load the LR (Landsat) input for one sample under the active `source` mode.

        Args:
            idx (int): File id used to locate the sample's image/feature
                file (i.e. `self.full_idxs[dataset_idx]`, not the dataset
                index itself).

        Returns:
            torch.Tensor | None:
                - ``source="features"``: cached pooled LR feature vector
                  loaded from ``self.lr_features_dir / f"image_{idx}.pt"``,
                  or `None` if `self.lr_features_dir` is unset (no
                  `lr_feature_run_name` given).
                - ``source="preprocessed"``: pre-transformed tensor loaded
                  from ``self.landsat_images_preprocessed / f"image_{idx}.pt"``,
                  further center-cropped and resized to ``_IMG_SIZE`` via
                  `_crop_resize_landsat` if `self.lr_crop_km` is set.
                - ``source="raw"``: 6-band GeoTIFF at
                  ``self.landsat_images / f"image_{idx}.tif"``, read via
                  rasterio, resized to ``(6, _IMG_SIZE, _IMG_SIZE)`` float32
                  with bilinear interpolation if `self.scale_to_img_size` is
                  True and the native size differs, then passed through
                  `self.transform_landsat`. If `self.scale_to_img_size` is
                  False, the tensor keeps its native on-disk resolution.
        """
        if self.source == "features":
            if self.lr_features_dir is None:
                return None
            return torch.load(self.lr_features_dir / f"image_{idx}.pt", weights_only=False)

        if self.source == "preprocessed":
            landsat_tensor = torch.load(
                self.landsat_images_preprocessed / f"image_{idx}.pt", weights_only=False
            )
            if self.lr_crop_km is not None:
                landsat_tensor = self._crop_resize_landsat(landsat_tensor)
            return landsat_tensor

        tif_path = self.landsat_images / f"image_{idx}.tif"

        with rasterio.open(tif_path) as src:
            data = src.read()  

        data = data.astype(np.float32)

        landsat_tensor = torch.from_numpy(data).float()

        S = self._IMG_SIZE
        if self.scale_to_img_size and (landsat_tensor.shape[1] != S or landsat_tensor.shape[2] != S):
            landsat_tensor = torch.nn.functional.interpolate(
                landsat_tensor.unsqueeze(0), size=(S, S), mode="bilinear", align_corners=False
            ).squeeze(0)

        if self.transform_landsat is not None:
            landsat_tensor = self.transform_landsat(landsat_tensor)

        return landsat_tensor

    def _crop_resize_landsat(self, landsat_tensor):
        """Center-crop a stored full-res Landsat tensor to `lr_crop_km` and resize back to `_IMG_SIZE`.

        The full-res preprocessed set is stored at ``_LANDSAT_FULLRES_SIZE``
        px (normalized). This casts to fp32, takes the centered
        ``self._crop_px`` window (precomputed from `lr_crop_km` and the
        metadata-inferred footprint in `__init__`), and bilinearly resizes
        the crop to ``_IMG_SIZE``.

        Args:
            landsat_tensor (torch.Tensor): ``(6, _LANDSAT_FULLRES_SIZE,
                _LANDSAT_FULLRES_SIZE)`` full-resolution normalized Landsat
                tensor.

        Returns:
            torch.Tensor: ``(6, _IMG_SIZE, _IMG_SIZE)`` float32 tensor, the
            centered crop resized to the model's expected input size.

        Raises:
            ValueError: If `landsat_tensor` is not `_LANDSAT_FULLRES_SIZE`
                square.
        """
        landsat_tensor = landsat_tensor.float()
        _, H, W = landsat_tensor.shape
        if H != self._LANDSAT_FULLRES_SIZE or W != self._LANDSAT_FULLRES_SIZE:
            raise ValueError(
                f"lr_crop_km expects {self._LANDSAT_FULLRES_SIZE}x{self._LANDSAT_FULLRES_SIZE} "
                f"full-res tensors, got {H}x{W}."
            )

        crop = self._crop_px
        if crop < H:
            top = (H - crop) // 2
            left = (W - crop) // 2
            landsat_tensor = landsat_tensor[:, top:top + crop, left:left + crop]

        S = self._IMG_SIZE
        if landsat_tensor.shape[1] != S or landsat_tensor.shape[2] != S:
            landsat_tensor = torch.nn.functional.interpolate(
                landsat_tensor.unsqueeze(0), size=(S, S),
                mode="bilinear", align_corners=False,
            ).squeeze(0)
        return landsat_tensor

def collate_multiscale(batch):
    """DataLoader collate function for `FMoWMultiScaleDataset.__getitem__` samples.

    Stacks each per-sample tensor into a batch dimension, except for keys
    whose value is `None` for every sample in the batch (e.g. an unused
    branch under ``source="features"``), which are passed through as `None`
    so single-branch feature loading doesn't require dummy tensors.

    Args:
        batch (list[tuple[dict, torch.Tensor, torch.Tensor]]): List of
            ``(x, y, metadata)`` samples as returned by
            `FMoWMultiScaleDataset.__getitem__`.

    Returns:
        tuple[dict[str, torch.Tensor | None], torch.Tensor, torch.Tensor]:
        ``(x_batch, y_batch, metadata_batch)`` where `x_batch` has the same
        keys as each sample's `x`, each stacked to ``(batch_size, ...)`` (or
        `None`); `y_batch` is ``(batch_size,)``; and `metadata_batch` is
        ``(batch_size, num_metadata_fields)``.
    """
    xs, ys, metas = zip(*batch)
    keys = xs[0].keys()
    # A key is None for every sample (e.g. an unused branch in feature mode); keep
    # it None rather than stacking, so single-branch feature loading works.
    x_batch = {
        k: None if xs[0][k] is None else torch.stack([x[k] for x in xs], dim=0)
        for k in keys
    }
    return x_batch, torch.stack(ys, dim=0), torch.stack(metas, dim=0)


if __name__ == "__main__":
    dataset = FMoWMultiScaleDataset(
        fmow_dir="/home/henicke/data",
        landsat_dir="/home/datasets4/FMoW_LandSat",
        lr_extension_factor=3.0,
    )

    sample, y, metadata = dataset[271258]
    print("HR shape:", sample["rgb"].shape)
    print("LR shape:", sample["landsat"].shape)

    print(sample["rgb"].amin(dim=(1, 2)), sample["rgb"].amax(dim=(1, 2)))
    print(sample["landsat"].amin(dim=(1, 2)), sample["landsat"].amax(dim=(1, 2)))

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(((sample["rgb"][[2, 1, 0], :, :] + 1) / 2).permute(1, 2, 0))
    axes[0].set_title("FMoW High-Res")
    axes[1].imshow(((sample["landsat"][[2, 1, 0], :, :] + 1) / 2).permute(1, 2, 0))
    axes[1].set_title("Landsat Broad-Scale")
    plt.show()
