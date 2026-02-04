from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from torchvision import transforms
import rasterio
from wilds.datasets.wilds_dataset import WILDSDataset


class FMoWMultiScaleDataset(WILDSDataset):
    """
    Extended FMoW Dataset that loads both:
    1. RGB images (224x224, 3 channels) - high resolution FMoW samples
    2. Landsat GeoTIFF images (224x224, 6 bands) - broader scale context

    Inherits from WILDSDataset to maintain compatibility with WILDS package.
    """

    _dataset_name = "fmow_multiscale"

    def __init__(
        self,
        fmow_dir="data",
        landsat_dir="data",
        metadata_csv="rgb_metadata.csv",
        split_scheme="official",
        use_ood_val=True,
        seed=111,
        transform_rgb=None,
        transform_landsat=None,
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
        # Import base FMoW dataset to reuse split logic
        from wilds import get_dataset

        # Initialize base FMoW dataset to get splits and metadata
        self.base_dataset = get_dataset(
            dataset="fmow", root_dir=fmow_dir, download=False, split_scheme=split_scheme
        )

        # Store directories
        self.root_fmow = Path(fmow_dir) / f"fmow_v{self.base_dataset.version}"
        self.fmow_images = self.root_fmow / "images"
        self.root_landsat = Path(landsat_dir) / "fmow_landsat"
        self.landsat_images = self.root_landsat / "images"

        # Inherit all necessary attributes from base dataset
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

    def get_default_transform_rgb(self):
        """Default transform for RGB images (ImageNet normalization)"""
        return transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                # transforms.Normalize(
                #     mean=[0.485, 0.456, 0.406],
                #     std=[0.229, 0.224, 0.225]
                # )
            ]
        )

    def get_default_transform_landsat(self):
        """
        Default transform for Landsat images.
        Note: This returns a preprocessing function, not a torchvision transform.
        """
        return None  # Will handle normalization manually in load_landsat

    def __len__(self):
        return len(self._y_array)

    def __getitem__(self, idx):
        """
        Returns tuple of (x, y, metadata) following WILDS convention.
        x is now a dict with 'rgb' and 'landsat' keys.
        """
        # Get actual file index (accounting for sequestered images)
        file_idx = self.full_idxs[idx]

        # Load RGB image
        rgb_img = self.get_rgb_input(file_idx)

        # Load Landsat image (all 6 bands)
        landsat_img = self.get_landsat_input(file_idx)

        # Get label and metadata
        y = self._y_array[idx]
        metadata = self._metadata_array[idx]

        # Combine into dict for x
        x = {"rgb": rgb_img, "landsat": landsat_img}

        return x, y, metadata

    def get_rgb_input(self, idx):
        """Load RGB FMoW image"""
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
        tif_path = self.landsat_images / f"image_{idx}.tif"

        with rasterio.open(tif_path) as src:
            # Read all 6 bands
            data = src.read()  # Shape: (6, H, W)

        # Convert to float32
        data = data.astype(np.float32)

        # TODO
        # Normalize (adjust based on your Landsat data range)
        # Typical Landsat 8 surface reflectance ranges from 0-10000
        # Adjust this normalization based on your specific data
        # data = np.clip(data / 10000.0, 0, 1)

        # Resize to 224x224 if needed
        if data.shape[1] != 224 or data.shape[2] != 224:
            # Use PIL for resizing each band
            resized_bands = []
            for band_idx in range(data.shape[0]):
                band = data[band_idx]
                band_img = Image.fromarray((band * 255).astype(np.uint8))
                band_img = band_img.resize((224, 224), Image.BILINEAR)
                resized_bands.append(np.array(band_img) / 255.0)
            data = np.stack(resized_bands, axis=0)

        # Convert to tensor
        landsat_tensor = torch.from_numpy(data).float()

        # Apply custom normalization if provided
        if self.transform_landsat is not None:
            landsat_tensor = self.transform_landsat(landsat_tensor)

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


# Custom collate function for DataLoader
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
        fmow_dir="/home/erik/git/bigpicture/data",
        landsat_dir="/home/erik/git/bigpicture/data",
    )

    sample, y, metadata = dataset[269149]
    print("HR shape:", sample["rgb"].shape)
    print("LR shape:", sample["landsat"].shape)

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(sample["rgb"][[2, 1, 0], :, :].permute(1, 2, 0))
    axes[0].set_title("FMoW High-Res")
    axes[1].imshow(sample["landsat"][[2, 1, 0], :, :].permute(1, 2, 0))
    axes[1].set_title("Landsat Broad-Scale")
    plt.show()
