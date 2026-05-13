from pathlib import Path
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision.transforms import v2 as transforms
import rasterio
from wilds.datasets.wilds_dataset import WILDSDataset


class FMoWMultiScaleDataset(WILDSDataset):
    """
    Extended FMoW Dataset that loads both:
    1. FMoW RGB images (224x224, 3 channels) - high resolution FMoW samples
    2. Landsat GeoTIFF images (224x224, 6 bands) - broader scale context

    Inherits from WILDSDataset to maintain compatibility with WILDS package.
    """

    _dataset_name = "fmow_multiscale"

    _IMG_SIZE = 224

    def __init__(
        self,
        fmow_dir="data",
        landsat_dir="data",
        preprocessed_dir=None,
        extended_metadata_csv="rgb_metadata_extended.csv",
        split_scheme="official",
        use_ood_val=True,
        seed=111,
        transform_rgb=None,
        transform_landsat=None,
        augment=False,
        hflip_prob=0.5,
        vflip_prob=0.5,
        image_norm="fmow-statistics",
        spatial_coord_grid=False,
        spatial_overlap_mask=False,
        overlap_mask_type="binary",
        lr_span_km=None,
    ):
        """
        Args:
            root_dir: Root directory containing the dataset
            fmow_rgb_dir: Subdirectory with FMoW RGB images (relative to root_dir)
            landsat_dir: Subdirectory with Landsat GeoTIFF files (relative to root_dir)
            metadata_csv: CSV file with metadata (relative to root_dir)
            split_scheme: 'official' (time_after_2016) or other WILDS split schemes
            use_ood_val: Whether to use OOD validation set
            seed: Random seed
            transform_rgb: Transforms for RGB images
            transform_landsat: Transforms for Landsat images
        """
        from wilds import get_dataset

        # Initialize base FMoW dataset to get splits and metadata
        self.base_dataset = get_dataset(
            dataset="fmow", root_dir=fmow_dir, download=False, split_scheme=split_scheme
        )

        self.root_fmow = Path(fmow_dir) / f"fmow_v{self.base_dataset.version}"
        self.fmow_images = self.root_fmow / "images"
        self.root_landsat = Path(landsat_dir) / "fmow_landsat"
        self.landsat_images = self.root_landsat / "images"
        if preprocessed_dir is not None:
            self.fmow_images_preprocessed = Path(preprocessed_dir) / "fmow_preprocessed" / "fmow_rgb"
            self.landsat_images_preprocessed = Path(preprocessed_dir) / "fmow_preprocessed" / "landsat"
        self.image_norm = image_norm

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

        # Transforms
        self.transform_rgb = transform_rgb or self.get_default_transform_rgb()
        self.transform_landsat = (
            transform_landsat or self.get_default_transform_landsat()
        )

        self.augment = augment
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob

        self.spatial_coord_grid = spatial_coord_grid
        self.spatial_overlap_mask = spatial_overlap_mask
        self.overlap_mask_type = overlap_mask_type
        self.lr_span_km = lr_span_km

        if spatial_coord_grid or spatial_overlap_mask:
            if lr_span_km is None:
                raise ValueError("lr_span_km is required when spatial features are enabled.")
            S = self._IMG_SIZE
            self._lr_res = lr_span_km * 1000.0 / S
            pixel = torch.arange(S, dtype=torch.float32)
            py, px = torch.meshgrid(pixel * self._lr_res, pixel * self._lr_res, indexing="ij")
            self._coord_grid_lr = torch.stack([px, py], dim=0)  # (2, S, S)


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
            tensors["coord_grid_lr"] = self._coord_grid_lr  # (2, S, S), shared
            pixel = torch.arange(S, dtype=torch.float32)
            py, px = torch.meshgrid(pixel * hr_res + offset, pixel * hr_res + offset, indexing="ij")
            tensors["coord_grid_hr"] = torch.stack([px, py], dim=0)

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

        if self.augment:
            rgb_img, landsat_img, spatial = self._apply_augmentation(rgb_img, landsat_img, spatial)

        x = {
            "rgb": rgb_img,
            "landsat": landsat_img,
            "coords": coords,
            "img_span_km": self._metadata_array[idx, img_span_idx],
            **spatial,
        }

        return x, y, metadata

    def get_rgb_input(self, idx):
        """Load RGB FMoW image"""

        if hasattr(self, 'fmow_images_preprocessed') and self.fmow_images_preprocessed is not None:
            return torch.load(self.fmow_images_preprocessed / f"rgb_img_{idx}.pt", weights_only=False)

        img_path = self.fmow_images / f"rgb_img_{idx}.png"
        img = Image.open(img_path).convert("RGB")

        if self.transform_rgb is not None:
            img = self.transform_rgb(img)

        return img

    def get_landsat_input(self, idx):
        """
        Load Landsat GeoTIFF image with all 6 bands.
        Returns tensor of shape (6, 224, 224).
        """
        if hasattr(self, 'landsat_images_preprocessed') and self.landsat_images_preprocessed is not None:
            return torch.load(self.landsat_images_preprocessed / f"image_{idx}.pt", weights_only=False)

        tif_path = self.landsat_images / f"image_{idx}.tif"

        with rasterio.open(tif_path) as src:
            data = src.read()  

        data = data.astype(np.float32)

        landsat_tensor = torch.from_numpy(data).float()

        if landsat_tensor.shape[1] != 224 or landsat_tensor.shape[2] != 224:
            landsat_tensor = torch.nn.functional.interpolate(
                landsat_tensor.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False
            ).squeeze(0)

        if self.transform_landsat is not None:
            landsat_tensor = self.transform_landsat(landsat_tensor)

        return landsat_tensor

    def get_input(self, idx):
        file_idx = self.full_idxs[idx]
        coords_start = self._metadata_fields.index("lat")
        coords = self._metadata_array[idx, coords_start:coords_start + 2]
        img_span_idx = self._metadata_fields.index("img_span_km")
        img_span_km = self._metadata_array[idx, img_span_idx]
        spatial = self._build_spatial_tensors(img_span_km.item())
        return {
            "rgb": self.get_rgb_input(file_idx),
            "landsat": self.get_landsat_input(file_idx),
            "coords": coords,
            "img_span_km": img_span_km,
            **spatial,
        }

    def eval(self, y_pred, y_true, metadata, prediction_fn=None):
        """Reuse evaluation from base FMoW dataset"""
        return self.base_dataset.eval(y_pred, y_true, metadata, prediction_fn)


def collate_multiscale(batch):
    xs, ys, metas = zip(*batch)
    keys = xs[0].keys()
    x_batch = {k: torch.stack([x[k] for x in xs], dim=0) for k in keys}
    return x_batch, torch.stack(ys, dim=0), torch.stack(metas, dim=0)


if __name__ == "__main__":
    dataset = FMoWMultiScaleDataset(
        fmow_dir="/home/henicke/data",
        landsat_dir="/home/datasets4/FMoW_LandSat",
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
