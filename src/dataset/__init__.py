"""Dataset package: PyTorch `Dataset` classes pairing FMoW and Landsat imagery.

Currently contains `fmow_multiscale_dataset`, which defines
`FMoWMultiScaleDataset` (a WILDS-compatible dataset serving co-located
FMoW high-resolution RGB and broader-scale Landsat images, either raw,
pre-transformed, or as cached encoder features) and `collate_multiscale`,
its matching `DataLoader` collate function.
"""
