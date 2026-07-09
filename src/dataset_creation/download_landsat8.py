"""
download_landsat8.py

Downloads Landsat imagery (Landsat 8, falling back to Landsat 5/7) from
Google Earth Engine for each FMoW sample in the WILDS [train, val, test]
splits, saving one GeoTIFF per sample under `IMAGES_DIR` and writing a
metadata CSV recording which collection/date-range was used for each
download.

Functions:
    main: Entry point; authenticates with Earth Engine, builds the Landsat
        image collections, selects the WILDS + not-yet-downloaded rows from
        the metadata CSV, and downloads them in parallel via `download_image`.
    download_image: Downloads a single sample's Landsat GeoTIFF, retrying on
        failure; used (via `functools.partial`) as the per-row worker for
        `pandarallel`'s `parallel_apply` in `main`.
    get_median_request / get_landsat_col: Build the median composite image
        request for a sample's region of interest, searching progressively
        wider date ranges and satellite collections until enough scenes are
        found.
    compute_img_span / extract_region_of_interest: Convert a sample's center
        coordinates and span (km) into a lon/lat rectangle (Earth Engine
        geometry).
    get_date_range / month_add: Build and widen the date window searched for
        each sample.
    mask_slc_gaps / filter_dates_and_roi / scale_optical_bands: Earth Engine
        image-collection preprocessing helpers (QA masking, date+ROI
        filtering, DN-to-reflectance scaling) used by `get_landsat_col` and
        `main`.
    get_fmow_wilds_mask / get_missing_mask: Row-selection helpers used by
        `main` to restrict the metadata to WILDS rows / rows not yet
        downloaded.
"""
import os
import pathlib
from functools import partial
import time
from datetime import datetime
from dateutil.relativedelta import relativedelta
import shutil
import random
import logging
import typing
import pytz
import re


import requests
import numpy as np
import pandas as pd
from pandarallel import pandarallel
import ee
from geopy.distance import geodesic
from geopy.point import Point

pandarallel.initialize(nb_workers=20)

PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent.resolve()
DATA_DIR = (PROJECT_ROOT.parent.parent.parent /
            "datasets4" / "FMoW_LandSat" / "fmow_landsat")
IMAGES_DIR = DATA_DIR / "images"
LOG_DIR = PROJECT_ROOT / "log"
LOG_FILE = LOG_DIR / 'download.log'
EE_PROJECT_NAME = 'seeing-the-big-picture'
APPEND_DOWNLOAD = True
EXTENSION_FACTOR = 3.0
SCALE = 30.0
MAX_PIXELS = 1e8
MIN_COL_SIZE = 50


def compute_img_span(img_center_lon: float, img_center_lat: float, img_span_km: float) -> tuple[float, float]:
    """Compute the distance in degrees longitude and latitude from kilometers.

    Args:
        img_center_lon (float): Longitude of the image center.
        img_center_lat (float): Latitude of the image center.
        img_span_km (float): Span of the image in kilometer.

    Returns:
        tuple[float, float]: Span of the image in degrees longitude and latitude.
    """
    img_center = Point(img_center_lat, img_center_lon)
    img_upper = geodesic(kilometers=(img_span_km / 2)
                         ).destination(img_center, 0)
    img_lower = geodesic(kilometers=(img_span_km / 2)
                         ).destination(img_center, 180)
    img_left = geodesic(kilometers=(img_span_km / 2)
                        ).destination(img_center, 90)
    img_right = geodesic(kilometers=(img_span_km / 2)
                         ).destination(img_center, 270)
    return (np.abs(img_left.longitude - img_right.longitude), np.abs(img_upper.latitude - img_lower.latitude))


def extract_region_of_interest(sample_metadata: pd.Series, span_km: float) -> ee.Geometry.Rectangle:
    """Extract region of interest (ROI) from the metadata.

    Args:
        sample_metadata (pd.Series): Metadata for fmow sample containing image coordinates.
        span_km (float): Determines the size of ROI in kilometers.

    Returns:
        ee.Geometry.Rectangle: Region of interest.
    """
    img_center_lon, img_center_lat = (sample_metadata.get(
        "img_center_lon"), sample_metadata.get("img_center_lat"))
    img_span_lon, img_span_lat = compute_img_span(
        img_center_lon, img_center_lat, span_km)
    extended_bounds = [img_center_lon - (img_span_lon / 2), img_center_lat - (img_span_lat / 2),
                       img_center_lon + (img_span_lon / 2), img_center_lat + (img_span_lat / 2)]
    return ee.Geometry.Rectangle(extended_bounds)


def get_date_range(sample_metadata, day_span=90):
    """Build a date range centered on a sample's acquisition timestamp.

    Args:
        sample_metadata (pd.Series): Metadata for one FMoW sample; must
            contain a `timestamp` field formatted as `%Y-%m-%dT%H:%M:%SZ`.
        day_span (int): Total width of the date range in days, centered on
            the sample's timestamp. Defaults to 90.

    Returns:
        tuple[str, str]: `(start_date, end_date)` as `%Y-%m-%d` strings.
    """
    sample_date = datetime.strptime(
        sample_metadata["timestamp"], '%Y-%m-%dT%H:%M:%SZ')
    start_date = sample_date - relativedelta(days=(day_span // 2))
    end_date = sample_date + relativedelta(days=(day_span // 2))
    return (start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))


def month_add(date: str, months_to_add=1) -> str:
    """Add (or subtract, if negative) whole months to a date string.

    Args:
        date (str): Date string in `%Y-%m-%d` format, e.g. "2020-02-01".
        months_to_add (int): Number of months to add; negative to subtract.
            Defaults to 1.

    Returns:
        str: The resulting date, formatted as `%Y-%m-%d`.
    """
    date_time_obj = datetime.strptime(date, '%Y-%m-%d')
    new_date_time_obj = date_time_obj + relativedelta(months=+months_to_add)
    new_date_str = new_date_time_obj.strftime('%Y-%m-%d')
    return new_date_str


def mask_slc_gaps(image):
    """Mask out flagged pixels using the Landsat `QA_PIXEL` band.

    Keeps only pixels where the low 6 bits of `QA_PIXEL` are all zero, i.e.
    pixels not flagged by any of Collection 2's fill/cloud/shadow/snow QA
    bits (this also removes Landsat 7 SLC-off scan-line gaps, which are
    flagged as fill).

    Args:
        image (ee.Image): Landsat surface reflectance image with a
            `QA_PIXEL` band.

    Returns:
        ee.Image: The same image with an updated mask hiding flagged pixels.
    """
    qa_pixel = image.select('QA_PIXEL')
    full_mask = qa_pixel.bitwiseAnd(63).eq(0)
    return image.updateMask(full_mask)


def filter_dates_and_roi(col, date_range, roi, contains_roi=True):
    """Filter an Earth Engine image collection by date range, ROI, and QA mask.

    Applies `mask_slc_gaps` to every image after filtering by date and
    bounding geometry. If `contains_roi` is True, further restricts to
    images whose footprint fully contains `roi`.

    Args:
        col (ee.ImageCollection): Landsat image collection to filter.
        date_range (tuple[str, str]): `(start_date, end_date)` as
            `%Y-%m-%d` strings.
        roi (ee.Geometry.Rectangle): Region of interest to filter/intersect against.
        contains_roi (bool): If True, only keep images whose footprint fully
            contains `roi`. Defaults to True.

    Returns:
        ee.ImageCollection: The filtered (and QA-masked) collection.
    """
    fcol = (col.filterDate(date_range[0], date_range[1])
            .filterBounds(roi)
            .map(mask_slc_gaps))
    if contains_roi:
        return fcol.filter(ee.Filter.contains('.geo', roi))
    return fcol


def get_landsat_col(cols, date_range: tuple, region_of_interest):
    """Find a Landsat collection with enough scenes covering the ROI, widening the date range if needed.

    Tries each collection in `cols` filtered to `date_range` and
    `region_of_interest`, first requiring images whose footprint fully
    contains the ROI (`contains_roi=True`), then relaxing to any
    intersecting image (`contains_roi=False`). Returns the first
    (collection, contains_roi) combination whose filtered size exceeds
    `MIN_COL_SIZE`. If none qualify, recurses with the date range widened by
    2 months on each side.

    Args:
        cols (dict[str, ee.ImageCollection]): Candidate Landsat collections
            keyed by name (e.g. "l8", "l5", "l7"), tried in dict order.
        date_range (tuple[str, str]): `(start_date, end_date)` as
            `%Y-%m-%d` strings.
        region_of_interest (ee.Geometry.Rectangle): Region of interest to
            filter against.

    Returns:
        tuple[ee.ImageCollection, tuple[str, str], str, int]:
            `(filtered_col, date_range, col_name, col_size)` for the
            collection/date range that satisfied `MIN_COL_SIZE`;
            `date_range` reflects any widening that occurred.
    """

    for contains_roi in [True, False]:
        for col_name, col in cols.items():
            filtered_col = filter_dates_and_roi(
                col, date_range, region_of_interest, contains_roi=contains_roi)
            size = filtered_col.size().getInfo()
            if size > MIN_COL_SIZE:
                return (filtered_col, date_range, col_name, size)

    new_date_0 = month_add(date_range[0], -2)
    new_date_1 = month_add(date_range[1], 2)
    return get_landsat_col(cols, (new_date_0, new_date_1), region_of_interest)


def get_median_request(sample_metadata, cols, cols_bands, span_km):
    """Build a median-composite Landsat image request for one sample's region of interest.

    Combines `extract_region_of_interest`, `get_date_range`, and
    `get_landsat_col` to find a suitable collection/date range, then takes
    the per-pixel median across the resulting collection, clips it to the
    ROI, and reprojects to EPSG:3857 at 30m resolution.

    Args:
        sample_metadata (pd.Series): Metadata for one FMoW sample containing
            image center coordinates, span, and timestamp.
        cols (dict[str, ee.ImageCollection]): Candidate Landsat collections
            keyed by name, passed through to `get_landsat_col`.
        cols_bands (dict[str, list[str]]): Optical band names per collection.
            Not used directly here; kept for a uniform signature with
            `download_image`, which selects bands from the result via `col_name`.
        span_km (float): Size of the region of interest to request, in kilometers.

    Returns:
        tuple[ee.Image, ee.Geometry.Rectangle, tuple[str, str], str, int]:
            `(median_img, region_of_interest, date_range, col_name, col_size)`.
    """
    region_of_interest = extract_region_of_interest(sample_metadata, span_km)
    col, date_range, col_name, col_size = get_landsat_col(
        cols, get_date_range(sample_metadata), region_of_interest)
    median_img = col.median().clip(region_of_interest).reproject(
        crs='EPSG:3857', scale=30)
    return (median_img, region_of_interest, date_range, col_name, col_size)


def download_image(sample_metadata: pd.Series, cols, cols_bands, span_km: float, pixel_dim: int, logger: logging.Logger):
    """Download the median-composite Landsat GeoTIFF for one FMoW sample from Google Earth Engine.

    Retries up to 30 times (combined across both steps) on any exception:
    first to build the median image request via `get_median_request`, then
    to fetch and save the GeoTIFF via its download URL to
    `IMAGES_DIR/image_<sample_idx>.tif`, where `sample_idx` is
    `sample_metadata.name` (the row's index/id).

    Args:
        sample_metadata (pd.Series): Metadata for one FMoW sample containing
            image center coordinates, span, and timestamp; `.name` is used
            as the sample id for the output filename.
        cols (dict[str, ee.ImageCollection]): Candidate Landsat collections
            keyed by name, passed through to `get_median_request`.
        cols_bands (dict[str, list[str]]): Optical band names per collection,
            used to select which bands to download once a collection is chosen.
        span_km (float): Size of the region of interest to download, in kilometers.
        pixel_dim (int): Width/height of the downloaded image, in pixels.
        logger (logging.Logger): Logger used to record retry attempts.

    Returns:
        pd.Series: `{"date_range": ..., "col_size": ..., "col": ...}`
            recording which date range, collection size, and collection name
            were used (each `None` if the request could never be built).
    """
    date_range, col_size, col_name = None, None, None
    attempts = 30
    attempts_left = attempts
    while attempts_left > 0:
        try:
            image_request, region, date_range, col_name, col_size = get_median_request(
                sample_metadata, cols, cols_bands, span_km
            )
        except Exception as e:
            logger.info(f"Attempt {attempts + 1 - attempts_left} - " + str(e))
            attempts_left -= 1
            time.sleep(1 + random.uniform(0, 1))
        else:
            break

    sample_idx = sample_metadata.name
    download_path = (
        IMAGES_DIR / f"image_{sample_idx}.tif")

    while attempts_left > 0:
        try:
            url = image_request.select(cols_bands[col_name]).getDownloadUrl({
                'scale': 30,
                'dimensions': f'{pixel_dim}x{pixel_dim}',
                'format': 'GeoTIFF',
                'filePerBand': False
            })
            response = requests.get(url, stream=True)

            if response.status_code != 200:
                raise response.raise_for_status()

            with open(download_path, 'wb') as out_file:
                shutil.copyfileobj(response.raw, out_file)

        except Exception as e:
            logger.info(f"Attempt {attempts + 1 - attempts_left} - " + str(e))
            attempts_left -= 1
            time.sleep(1 + random.uniform(0, 1))
        else:
            break

    return pd.Series({"date_range": date_range, "col_size": col_size, "col": col_name})


def scale_optical_bands(image, optical_bands):
    """Scale Landsat Collection 2 optical bands from DN to surface reflectance.

    Applies the standard `DN * 0.0000275 - 0.2` scaling factor (see
    https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LE07_C02_T1_L2#bands
    and .../LANDSAT_LC08_C02_T1_L2#bands) to the given bands and replaces
    them in place.

    Args:
        image (ee.Image): Landsat Collection 2 surface reflectance image.
        optical_bands (list[str]): Names of the optical bands to scale.

    Returns:
        ee.Image: The same image with `optical_bands` replaced by their
            scaled (reflectance) values.
    """
    scaled_optical_bands = (image
                            .select(optical_bands)
                            .multiply(0.0000275)
                            .add(-0.2))
    return image.addBands(scaled_optical_bands, optical_bands, True)


def get_fmow_wilds_mask(metadata):
    """Build a boolean mask selecting metadata rows that belong to the WILDS FMoW dataset.

    Mirrors the split construction in
    `download_preprocessed_subset.assign_wilds_split`: a row is included if
    it is a `train`/`val`/`test` row whose timestamp falls in the matching
    WILDS time window - `train`, `val`, or `test` within [2002, 2013) (ID
    period), `val` within [2013, 2016) (OOD-Val), or `test` within
    [2016, 2018) (OOD-Test). `split == "seq"` rows and any row outside these
    windows are excluded.

    Args:
        metadata (pd.DataFrame): Metadata with `timestamp` and `split` columns.

    Returns:
        pd.Series: Boolean mask, one entry per row of `metadata`, True for
            rows inside the WILDS dataset.
    """
    timestamps = pd.to_datetime(metadata['timestamp'])
    utc = pytz.UTC
    id_mask = (timestamps >= utc.localize(datetime(2002, 1, 1))) & (
        timestamps < utc.localize(datetime(2012 + 1, 1, 1)))
    ood_val_mask = (timestamps >= utc.localize(datetime(2013, 1, 1))) & (
        timestamps < utc.localize(datetime(2015 + 1, 1, 1)))
    ood_test_mask = (timestamps >= utc.localize(datetime(2016, 1, 1))) & (
        timestamps < utc.localize(datetime(2017 + 1, 1, 1)))
    train_mask = (metadata['split'] == 'train')
    val_mask = (metadata['split'] == 'val')
    test_mask = (metadata['split'] == 'test')
    return (id_mask & train_mask) | (ood_val_mask & val_mask) | (ood_test_mask & test_mask) | (id_mask & val_mask) | (id_mask & test_mask)


def get_missing_mask(metadata):
    """Find metadata rows whose Landsat GeoTIFF has not yet been downloaded.

    Reads all `.tif` filenames in `IMAGES_DIR`, extracts their sample
    indices (`image_<idx>.tif`), and compares them against `metadata`'s index.

    Args:
        metadata (pd.DataFrame): Metadata indexed by sample id (the same ids
            used in `image_<idx>.tif` filenames).

    Returns:
        np.ndarray: Boolean mask, one entry per row of `metadata` (in index
            order), True where that row's GeoTIFF is missing from `IMAGES_DIR`.
    """
    # Read all TIFF filenames
    tiff_files = [f for f in os.listdir(IMAGES_DIR) if f.endswith('.tif')]

    # Extract indices using regex: image_(\d+).tif -> \d+
    downloaded_indices = set()
    pattern = r'image_(\d+)\.tif'
    for filename in tiff_files:
        match = re.search(pattern, filename)
        if match:
            downloaded_indices.add(int(match.group(1)))

    # Get indices from metadata
    metadata_indices = metadata.index.astype(
        int).values

    missing_mask = ~np.isin(metadata_indices, list(downloaded_indices))
    return missing_mask


def main():
    """Download Landsat imagery for all not-yet-downloaded WILDS FMoW samples.

    Authenticates with Google Earth Engine, builds the Landsat 5/7/8 image
    collections (scaled to reflectance via `scale_optical_bands`), restricts
    `rgb_metadata_extended.csv` to WILDS rows (`get_fmow_wilds_mask`) that
    are not already downloaded (`get_missing_mask`, when `APPEND_DOWNLOAD`
    is True), computes a shared download span/pixel size from the largest
    selected sample's `img_span_km` times `EXTENSION_FACTOR`, then downloads
    each selected sample in parallel via `download_image` (through
    `pandarallel`). Writes the resulting metadata (augmented with the
    download's date range/collection info) to `rgb_metadata_download.csv`.

    Raises:
        NotADirectoryError: If `PROJECT_ROOT` or `DATA_DIR` does not exist.
        Exception: Re-raises whatever error `ee.Authenticate()` /
            `ee.Initialize()` raise, after printing an authentication hint.
    """
    if not (os.path.exists(PROJECT_ROOT) and os.path.exists(DATA_DIR)):
        raise NotADirectoryError()

    if not os.path.exists(IMAGES_DIR):
        os.makedirs(IMAGES_DIR)

    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    # Setup access for Google Earth Engine
    try:
        ee.Authenticate()
        ee.Initialize(project=EE_PROJECT_NAME)
    except Exception as e:
        print("Please authenticate Earth Engine: earthengine authenticate")
        raise e

    # Initialize logging
    logger = logging.getLogger('download')
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(LOG_FILE, mode='w')
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    # Setup Image collections
    optical_bands_l5_l7 = ['SR_B1', 'SR_B2',
                           'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7']
    optical_bands_l8 = ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7']
    scale_l5_l7 = partial(scale_optical_bands,
                          optical_bands=optical_bands_l5_l7)
    scale_l8 = partial(scale_optical_bands, optical_bands=optical_bands_l8)

    l5 = (ee.ImageCollection("LANDSAT/LT05/C02/T1_L2").map(scale_l5_l7))
    l7 = (ee.ImageCollection("LANDSAT/LE07/C02/T1_L2").map(scale_l5_l7))
    l8 = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2').map(scale_l8))

    cols = {'l8': l8, 'l5': l5, 'l7': l7}
    cols_bands = {'l8': optical_bands_l8,
                  'l7': optical_bands_l5_l7, 'l5': optical_bands_l5_l7}

    # Read metadata
    metadata = pd.read_csv(
        DATA_DIR / "rgb_metadata_extended.csv")

    wilds_mask = get_fmow_wilds_mask(metadata)
    missing_mask = get_missing_mask(metadata)
    metadata_mask = (
        wilds_mask & missing_mask) if APPEND_DOWNLOAD else wilds_mask
    metadata_selected = metadata.loc[metadata_mask]
    size = len(metadata_selected)

    max_span = metadata_selected["img_span_km"].max()
    download_span = max_span * EXTENSION_FACTOR
    pixel_dim = int(download_span * 1000 // 30)
    logger.info("Download span of %f km was used (%d times the biggest sample span).",
                download_span, EXTENSION_FACTOR)

    download_l8_image = partial(
        download_image, cols=cols, cols_bands=cols_bands, span_km=download_span, pixel_dim=pixel_dim, logger=logger)

    start = time.time()
    download_metadata = metadata_selected.parallel_apply(
        download_l8_image, axis=1)
    end = time.time()
    logger.info("Download of %d images took %s seconds.",
                size, f"{end - start:.2f}")
    print(f"Download of {size} images took {end - start:.2f} seconds.")

    metadata_downloaded = pd.concat(
        [metadata_selected, download_metadata], axis=1)
    metadata_downloaded.to_csv(DATA_DIR / "rgb_metadata_download.csv")


if __name__ == '__main__':
    main()
