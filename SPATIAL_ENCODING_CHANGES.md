# Spatial Encoding Implementation — Change Documentation

## Overview

This implementation adds three orthogonal, independently toggleable spatial encoding approaches that give the model explicit signals about the spatial relationship between HR (RGB) and LR (Landsat) images. All approaches are disabled by default, preserving full backward compatibility.

## New File

### `src/models/components/spatial_encoding.py`

Defines `SpatialEncoding(nn.Module)` — a Fourier feature encoding module.

- **Constructor** accepts `fourier_bands` (number of frequency bands L) and `fourier_proj_dim` (output channels after 1×1 conv projection).
- Precomputes frequency buffer: `π · 2^i` for `i ∈ [0, L)`.
- **`forward(coord_grid)`** takes a `(B, 2, H, W)` coordinate grid in meters, computes `[sin(f·x), cos(f·x), sin(f·y), cos(f·y)]` for each frequency → `(B, 4L, H, W)`, then projects via a learned `Conv2d(4L → proj_dim, 1×1)`.
- **`extra_channels`** property returns `fourier_proj_dim` for channel counting.

HR and LR receive **separate instances** with independent weights. Because the input coordinate grids differ (HR covers a small area, LR covers the full extent), the Fourier features will have different oscillation patterns at the same frequency, giving the model a direct scale signal.

## Modified Files

### `src/models/components/branches.py`

**`Branch` base class (line 17):**
- Relaxed `in_channels` validation from `3 ≤ in_channels ≤ 6` to `in_channels ≥ 3`. This allows higher channel counts when spatial features are concatenated.

**`DualBranch`:**
- Added constructor params: `coord_channels: bool`, `hr_spatial_encoding: Optional[SpatialEncoding]`, `lr_spatial_encoding: Optional[SpatialEncoding]`.
- Constructor validates that encoder `in_channels` match the expected count (base channels + spatial extras).
- `forward()` now:
  1. **(Approach 1)** If `self.coord_channels` is true, appends raw `coord_grid_hr` and `coord_grid_lr` from the `x` dict (+2 channels each).
  2. **(Approach 2)** If `"overlap_mask"` key exists in `x`, appends it to LR (+1 channel).
  3. **(Approach 3)** If spatial encoding modules are present and coord grids exist, appends Fourier-projected features (+`proj_dim` channels each).
  4. Concatenates extras onto image tensors before passing to encoders.

The `coord_channels` flag is separate from the coord grid tensors being present — Fourier encoding also needs coord grids as input but should not add raw coordinate channels unless explicitly requested.

### `src/dataset/fmow_multiscale_dataset.py`

**Constructor:**
- Added params: `spatial_coord_grid`, `spatial_overlap_mask`, `overlap_mask_type` ("binary"/"gaussian"), `lr_extension_factor`.
- When spatial features are enabled, computes `lr_span_km = max(img_span_km) * lr_extension_factor` from the dataset metadata, then precomputes the constant LR coordinate grid `(2, 224, 224)` in meters from the LR image top-left.
- Added class constant `_IMG_SIZE = 224`.

**`_build_spatial_tensors(img_span_km)` (new method):**
- Builds per-sample spatial tensors based on the sample's `img_span_km`:
  - **Coord grids:** LR grid is constant (precomputed). HR grid is per-sample, computed from `hr_res = img_span_km * 1000 / 224` with an offset that aligns HR center to LR center. Both are in meters from LR top-left.
  - **Overlap mask:** Binary (1 where HR overlaps LR in normalized space) or Gaussian (smooth falloff from center). Shape `(1, 224, 224)`.

**`_apply_augmentation`:**
- Extended to accept and flip an optional dict of spatial tensors alongside images. Horizontal/vertical flips are applied consistently to all tensors.

**`__getitem__` / `get_input`:**
- Call `_build_spatial_tensors` and merge results into the `x` dict.

**`collate_multiscale`:**
- Genericized: now iterates over all keys in the `x` dict and stacks them, instead of listing keys explicitly. This makes it automatically handle any spatial tensors present in the dict.

### `src/train/utils.py`

**`make_multiscale_dataset`:**
- Added passthrough params: `spatial_coord_grid`, `spatial_overlap_mask`, `overlap_mask_type`, `lr_span_km`.

### `src/train/run_experiment.py`

**`_parse_spatial_cfg(cfg)` (new function):**
- Extracts `model.spatial_encoding` config (all keys default to disabled/zero).
- Computes `hr_extra` and `lr_extra` channel counts, `needs_coord_grid`, `needs_overlap_mask`, and `use_fourier`.
- Returns a dict used by both `make_model` and `make_data_loaders`.

**`make_data_loaders`:**
- Calls `_parse_spatial_cfg` to derive dataset flags and passes `spatial_coord_grid`, `spatial_overlap_mask`, `overlap_mask_type`, `lr_span_km` to `make_multiscale_dataset`.

**`make_model`:**
- For `DualBranch`-based models: computes dynamic `in_channels` as `base + spatial extras` and overrides the YAML values when calling `instantiate`. Creates `SpatialEncoding` instances if Fourier is enabled. Passes `coord_channels`, `hr_spatial_encoding`, `lr_spatial_encoding` to `DualBranch`.
- For non-`DualBranch` models (e.g., `CoordDualBranch`): original instantiation path preserved.

### `src/train/configs/model/densenet_concat.yaml`

Added `spatial_encoding` block with all approaches disabled by default:
```yaml
spatial_encoding:
  coord_channels: false       # Approach 1: raw coordinate grids (+2ch each)
  overlap_mask: false          # Approach 2: HR-LR overlap mask (+1ch on LR)
  overlap_mask_type: binary    # "binary" or "gaussian"
  fourier_bands: 0             # Approach 3: frequency bands (0 = disabled)
  fourier_proj_dim: 0          # 1×1 conv output dim (0 = disabled)
  lr_extension_factor: 3.0     # LR extent = max(img_span_km) * factor
```

Other DualBranch configs (film, d3g, multsim, geoprior) work without this block — `_parse_spatial_cfg` defaults everything to disabled.

## Channel Count Reference

| Active approaches         | HR in_channels | LR in_channels |
|---------------------------|----------------|----------------|
| None (default)            | 3              | 6              |
| Coord only                | 5              | 8              |
| Mask only                 | 3              | 7              |
| Fourier (L=3, proj=4)     | 7              | 10             |
| Coord + Mask + Fourier    | 9              | 13             |

## Coordinate System

Both HR and LR grids use a shared physical reference frame: **meters from the top-left of the LR image**.

- LR resolution: `lr_span_km * 1000 / 224` m/pixel (constant)
- HR resolution: `img_span_km * 1000 / 224` m/pixel (per-sample)
- HR offset: `112 * lr_res - 112 * hr_res` (aligns centers)

At the center pixel (112, 112), HR and LR coordinates are identical. HR values cluster in a narrow band around the center of LR's range — this reflects the correct physical relationship where the HR image covers a small subset of the LR extent.
