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
    """Host-specific directory holding the preprocessed and feature datasets."""
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
    if preprocessed_dir is None:
        return None
    return f"{_host_data_root()}/{preprocessed_dir}"


def resolve_feature_dir(run_name: str, run_idx: int, branch_subdir: str) -> Path:
    """Locate cached features on the current host (written by extract_features.py).

    Mirrors resolve_preprocessed_dir's host map:
    ``<data-root>/FMoW_LandSat_<run_name>_Features/run<run_idx>/fmow_features/<branch_subdir>``,
    where ``branch_subdir`` is ``fmow_rgb`` (HR) or ``landsat`` (LR).
    """
    return (
        Path(_host_data_root())
        / f"FMoW_LandSat_{run_name.title()}_Features"
        / f"run{run_idx}"
        / "fmow_features"
        / branch_subdir
    )


class FMoWMultiScaleDataset(WILDSDataset):
    """
    Extended FMoW Dataset that loads both:
    1. FMoW RGB images (224x224, 3 channels) - high resolution FMoW samples
    2. Landsat GeoTIFF images (224x224, 6 bands) - broader scale context

    Inherits from WILDSDataset to maintain compatibility with WILDS package.
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
        """
        Args:
            split_scheme: 'official' (time_after_2016) or other WILDS split schemes
            transform_rgb: Transforms for RGB images
            transform_landsat: Transforms for Landsat images
            scale_to_img_size: If True (default), Landsat images are resized to
                ``_IMG_SIZE`` (224). If False, they are kept at their native
                resolution (e.g. 497x497) — still normalized, just not downscaled.
                Only valid with ``source="raw"`` (the preprocessed/feature paths
                serve already-sized tensors); raises otherwise. Used by the
                full-res preprocessing step for the spatial-extent ablation.
            lr_crop_km: If set, center-crop the stored full-res Landsat tensor to
                this spatial extent (km) and resize the crop back to ``_IMG_SIZE``.
                Only valid with ``source="preprocessed"`` pointing at the full-res
                set (normalized 497x497). Must lie in ``(0, full_span]`` where the
                full footprint is inferred from metadata (max HR span x
                ``lr_extension_factor``), so ``lr_extension_factor`` is required.
                This is the knob for the spatial-extent ablation: a smaller value
                keeps a tighter centered patch (then up/downsampled to 224). When
                spatial features are on, it also defines ``lr_span_km`` so the coord
                grid / overlap mask track the cropped footprint.
            source: Which input mode to serve (mutually exclusive):
                - ``"raw"``: read raw images from ``fmow_dir`` (HR) and
                  ``landsat_dir`` (LR); transforms applied on the fly.
                - ``"preprocessed"``: read pre-transformed tensors from the single
                  ``preprocessed_dir`` (``fmow_preprocessed/{fmow_rgb,landsat}``).
                - ``"features"``: read cached encoder features (written by
                  extract_features.py); see ``hr_feature_run_name``/``lr_feature_run_name``.
                Only the dirs for the active mode are read; the others are ignored.
            preprocessed_dir: Base dir name for ``source="preprocessed"`` (required
                there); host-resolved internally via resolve_preprocessed_dir, so
                pass the config name (e.g. ``"FMoW_LandSat"``), not an absolute path.
            hr_feature_run_name: For ``source="features"``: run-name of the cached HR
                features (the ``<run>`` in ``FMoW_LandSat_<hr_feature_run_name>_Features``). When
                set, ``x["rgb"]`` is the precomputed HR feature vector.
            lr_feature_run_name: As ``hr_feature_run_name`` for the LR/landsat branch
                (``x["landsat"]``).
            feature_run_idx: For ``source="features"``: index of the rerun to load
                (0/1/2 -> run0/run1/run2). Required there. Leaving one of the two
                run-names ``None`` yields a ``None`` tensor for that branch (e.g.
                retraining a single-branch classifier on the other modality).
            leave_asia_out: If True, turn the WILDS split scheme into a geographic
                leave-one-continent-out split while keeping the split names intact:
                Asia samples are dropped from ``train``/``id_val``/``val`` and
                non-Asia samples are dropped from ``id_test``/``test`` (set to the
                ``-1`` unused code). The model then trains and is model-selected on
                non-Asia and is evaluated exclusively on the unseen Asia continent.
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
        """Default transform for RGB images."""
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
        """
        Default transform for Landsat images.

        See https://developers.google.com/earth-engine/datasets/catalog/landsat/ for scaling info.
        GeoTIFFs store reflectance values directly; stats computed over the full dataset.
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
        """Apply the same spatial augmentation to images and optional spatial tensors."""
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
        return len(self._y_array)

    def __getitem__(self, idx):
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
        """Load the HR input for the active ``source`` mode.

        features: cached HR feature vector (or None if no HR run was given).
        preprocessed: pre-transformed RGB tensor. raw: RGB image + transforms.
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
        """Load the LR input for the active ``source`` mode.

        features: cached LR feature vector (or None if no LR run was given).
        preprocessed: pre-transformed LR tensor. raw: Landsat GeoTIFF (6, 224, 224).
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
        """Center-crop a stored full-res Landsat tensor to ``lr_crop_km`` and
        resize the crop back to ``_IMG_SIZE``.

        The full-res set is normalized fp16 at ``_LANDSAT_FULLRES_SIZE`` px. We
        cast to fp32, take the centered ``_crop_px`` window, and resize
        down (or plain bilinear up) to 224. ``_crop_px`` is precomputed from
        ``lr_crop_km`` and the metadata-inferred footprint in ``__init__``.
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

    def get_input(self, idx):
        file_idx = self.full_idxs[idx]
        coords_start = self._metadata_fields.index("lat")
        coords = self._metadata_array[idx, coords_start:coords_start + 2]
        img_span_idx = self._metadata_fields.index("img_span_km")
        img_span_km = self._metadata_array[idx, img_span_idx]
        spatial = self._build_spatial_tensors(img_span_km.item())
        region_idx = self._metadata_fields.index("region")
        return {
            "rgb": self.get_rgb_input(file_idx),
            "landsat": self.get_landsat_input(file_idx),
            "coords": coords,
            "domain": self._metadata_array[idx, region_idx],
            "img_span_km": img_span_km,
            **spatial,
        }

    def eval(self, y_pred, y_true, metadata, prediction_fn=None):
        """Reuse evaluation from base FMoW dataset"""
        return self.base_dataset.eval(y_pred, y_true, metadata, prediction_fn)


def collate_multiscale(batch):
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
