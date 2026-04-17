from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
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

    def __init__(
        self,
        fmow_dir="data",
        landsat_dir="data",
        preprocessed_dir=None,
        metadata_csv="rgb_metadata.csv",
        split_scheme="official",
        use_ood_val=True,
        seed=111,
        transform_rgb=None,
        transform_landsat=None,
        augment=False,  
        hflip_prob=0.5,               
        vflip_prob=0.5,               
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

        # Transforms
        self.transform_rgb = transform_rgb or self.get_default_transform_rgb()
        self.transform_landsat = (
            transform_landsat or self.get_default_transform_landsat()
        )

        self.augment = augment
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
 

    def get_default_transform_rgb(self):
        """Default transform for RGB images."""
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

    def get_default_transform_landsat(self):
        """
        Default transform for Landsat images.

        See https://developers.google.com/earth-engine/datasets/catalog/landsat/ for scaling info.
        GeoTIFFs store reflectance values directly; stats computed over the full dataset.

        # Theoretical scaling from raw DN (0–65353) to reflectance:
        # landsat_scale = 2.75e-05; landsat_offset = -0.2
        # lower = 0 * landsat_scale + landsat_offset = -0.2
        # upper = 65353 * landsat_scale + landsat_offset ≈ 1.597
        # mean = (upper + lower) / 2 ≈ 0.699; std = (upper - lower) / 2 ≈ 0.899
        """
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

    def _apply_augmentation(self, rgb_img: torch.Tensor, landsat_img: torch.Tensor):
        """
        Apply the same spatial augmentation to both modalities to keep alignment.
        Expects tensors shaped:
          rgb_img:      (3, H, W)
          landsat_img:  (C, H, W)
        """
        if torch.rand(1).item() < self.hflip_prob:
            rgb_img = torch.flip(rgb_img, dims=[2])       # flip width
            landsat_img = torch.flip(landsat_img, dims=[2])

        if torch.rand(1).item() < self.vflip_prob:
            rgb_img = torch.flip(rgb_img, dims=[1])       # flip height
            landsat_img = torch.flip(landsat_img, dims=[1])

        return rgb_img, landsat_img

    def __len__(self):
        return len(self._y_array)

    def __getitem__(self, idx):
        """
        Returns tuple of (x, y, metadata) following WILDS convention.
        x is now a dict with 'rgb' and 'landsat' keys.
        """
        file_idx = self.full_idxs[idx]

        rgb_img = self.get_rgb_input(file_idx)
        landsat_img = self.get_landsat_input(file_idx)

        if self.augment:
            rgb_img, landsat_img = self._apply_augmentation(rgb_img, landsat_img)

        y = self._y_array[idx]
        metadata = self._metadata_array[idx]
        x = {"rgb": rgb_img, "landsat": landsat_img}

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

        if self.transform_landsat is not None:
            landsat_tensor = self.transform_landsat(landsat_tensor)

        # Resize to 224x224
        if landsat_tensor.shape[1] != 224 or landsat_tensor.shape[2] != 224:
            resized_bands = []
            for band_idx in range(landsat_tensor.shape[0]):
                band = landsat_tensor[band_idx, :, :]
                # Scale from [-1, 1] to [0, 1] for PIL
                band_scaled = (band + 1) / 2
                band_img = transforms.ToPILImage()(band_scaled)
                band_img = band_img.resize((224, 224), Image.BILINEAR)
                # Scale back from [0, 1] to [-1, 1]
                # v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]) is equivalent to transforms.ToTensor()
                band_resized = transforms.Compose([
                    transforms.ToImage(),
                    transforms.ToDtype(torch.float32, scale=True)
                ])(band_img).squeeze(0) * 2 - 1
                resized_bands.append(band_resized)
            landsat_tensor = torch.stack(resized_bands, dim=0)

        return landsat_tensor

    def get_input(self, idx):
        """
        Required by WILDSDataset interface.
        Returns the input for a given idx.
        """
        file_idx = self.full_idxs[idx]
        return {
            "rgb": self.get_rgb_input(file_idx),
            "landsat": self.get_landsat_input(file_idx),
        }

    def eval(self, y_pred, y_true, metadata, prediction_fn=None):
        """Reuse evaluation from base FMoW dataset"""
        return self.base_dataset.eval(y_pred, y_true, metadata, prediction_fn)


def collate_multiscale(batch):
    """
    Custom collate function to handle dict inputs from FMoWMultiScaleDataset.

    Args:
        batch: List of tuples (x, y, metadata) where x is dict with 'rgb' and 'landsat'

    Returns:
        tuple of (x_dict, y_batch, metadata_batch)
        where x_dict = {'rgb': tensor, 'landsat': tensor}
    """
    rgb_list = []
    landsat_list = []
    y_list = []
    metadata_list = []

    for x, y, metadata in batch:
        rgb_list.append(x["rgb"])
        landsat_list.append(x["landsat"])
        y_list.append(y)
        metadata_list.append(metadata)

    return (
        {
            "rgb": torch.stack(rgb_list, dim=0),
            "landsat": torch.stack(landsat_list, dim=0),
        },
        torch.stack(y_list, dim=0),
        torch.stack(metadata_list, dim=0),
    )


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
