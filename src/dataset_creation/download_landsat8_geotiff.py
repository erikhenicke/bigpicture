"""
download_landsat8.py

This script downloads landsat8 satellite imagery for each fmow sample (WILDS version, [train, val, test]).
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
LOG_FILE = PROJECT_ROOT / 'download.log'
EE_PROJECT_NAME = 'seeing-the-big-picture'
EXTENSION_FACTOR = 3.0
SCALE = 30.0
MAX_PIXELS = 1e8
MIN_COL_SIZE = 100


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


def mask_clouds_compute_validity(image: ee.Image, region_of_interest: ee.Geometry.Rectangle) -> ee.Image:
    """Updates an image mask, to filter out cloudy pixels and compute the fraction of non-masked pixels in the ROI.

    For a detailed description of the 'QA_PIXEL' flags see:
        https://www.usgs.gov/landsat-missions/landsat-collection-2-quality-assessment-bands

    Args:
        image (ee.Image): Image to filter clouds from.
        region_of_interest (ee.Geometry.Rectangle): Region to compute the fraction of non-masked pixels of the image for.
    Returns:
        ee.Image: Image that also holds the fraction of pixels, which are not masked.
    """
    qa = image.select('QA_PIXEL')
    # Only pixels for which the first 5 bits equal zero are not masked away.
    cloud_mask = qa.bitwiseAnd(int('11111', 2)).eq(0)
    masked_image = image.updateMask(cloud_mask)

    validity = ee.Number(
        masked_image.select(['SR_B4', 'SR_B3', 'SR_B2'])
        .mask()
        .reduce(ee.Reducer.min())
        .reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region_of_interest,
            scale=SCALE,
            maxPixels=MAX_PIXELS)
        .get('min')
    )
    return typing.cast(ee.Image, masked_image.set('validity', validity))


def get_requests(sample_metadata: pd.Series, l8: ee.ImageCollection, span_km: float) -> tuple[ee.Image, ee.Geometry.Rectangle, bool, str]:
    """Get request for least cloudy image and request for the infromation if a composite will be returned.

    Mosaic composite is requested, if single least cloudy image contains more than 1% of invalid pixels.

    Args:
        sample_metadata (pd.Series): Metadata for fmow sample containing image coordinates and span.
        l8 (ee.ImageCollection): Landsat8 image collection.
        span_km (float): Image size in kilometer to download.

    Returns:
        tuple[ee.Image, ee.Geometry.Rectangle, bool, str]: 
            - Request for least cloudy image
            - Region of interes
            - Bool that says if a composite will be returned
            - String date of the image, if it is not a composite.
    """
    date = None
    region_of_interest = extract_region_of_interest(sample_metadata, span_km)
    l8_cloud_masked = (l8
                       .filterBounds(region_of_interest)
                       .filterMetadata('CLOUD_COVER', 'less_than', 50)
                       .map(lambda img:
                            mask_clouds_compute_validity(
                                img, region_of_interest))
                       )
    least_cloudy = l8_cloud_masked.sort('validity', False).first()
    validity = ee.Number(least_cloudy.get('validity')).getInfo()
    is_least_cloudy_ok = validity >= 0.99
    if is_least_cloudy_ok:
        date = least_cloudy.date().format(None, 'GMT').getInfo()
    mosaic = l8_cloud_masked.mosaic()
    optical_bands = ['SR_B4', 'SR_B3', 'SR_B2']
    vis_params = {'bands': optical_bands, 'min': 0, 'max': 0.3}
    final_image = least_cloudy.visualize(
        **vis_params) if is_least_cloudy_ok else mosaic.visualize(**vis_params)
    return (final_image, region_of_interest, is_least_cloudy_ok, date)


def get_date_range(sample_metadata, day_span=90):
    sample_date = datetime.strptime(
        sample_metadata["timestamp"], '%Y-%m-%dT%H:%M:%SZ')
    start_date = sample_date - relativedelta(days=(day_span // 2))
    end_date = sample_date + relativedelta(days=(day_span // 2))
    return (start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))


def month_add(date: str, months_to_add=1) -> str:
    ''' date should be string in `'2020-02-01' format`

    Usage
        `day_add('2020-12-30',days_to_add = 2)`

    '''
    date_time_obj = datetime.strptime(date, '%Y-%m-%d')
    new_date_time_obj = date_time_obj + relativedelta(months=+months_to_add)
    new_date_str = new_date_time_obj.strftime('%Y-%m-%d')
    return new_date_str


def mask_slc_gaps(image):
    qa_pixel = image.select('QA_PIXEL')
    full_mask = qa_pixel.bitwiseAnd(63).eq(0)
    return image.updateMask(full_mask)


def filter_dates_and_roi(col, date_range, roi, contains_roi=True):
    fcol = (col.filterDate(date_range[0], date_range[1])
            .filterBounds(roi)
            .map(mask_slc_gaps))
    if contains_roi:
        return fcol.filter(ee.Filter.contains('.geo', roi))
    return fcol


def get_landsat_col(cols, date_range: tuple, region_of_interest):

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
    region_of_interest = extract_region_of_interest(sample_metadata, span_km)
    col, date_range, col_name, col_size = get_landsat_col(
        cols, get_date_range(sample_metadata), region_of_interest)
    median_img = col.median().clip(region_of_interest).reproject(
        crs='EPSG:3857', scale=30)
    return (median_img, region_of_interest, date_range, col_name, col_size)


def download_image(sample_metadata: pd.Series, cols, cols_bands, span_km: float, pixel_dim: int, logger: logging.Logger):
    """Downloads Landsat8 image from Google Earth Engine for the image coordinates of the given sample.

    Args:
        sample_metadata (pd.Series): Metadata for fmow sample containing image coordinates and span.
        l7 (ee.ImageCollection): Landsat7 image collection.
        l8 (ee.ImageCollection): Landsat8 image collection.
        span_km (float): Image size in kilometer to download.
        logger (logging.Logger):
    """
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
    """Scale factors taken from:
        https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LE07_C02_T1_L2#bands
        https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C02_T1_L2#bands
    """
    scaled_optical_bands = (image
                            .select(optical_bands)
                            .multiply(0.0000275)
                            .add(-0.2))
    return image.addBands(scaled_optical_bands, optical_bands, True)


def get_fmow_wilds_mask(metadata):
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


def main():
    if not (os.path.exists(PROJECT_ROOT) and os.path.exists(DATA_DIR)):
        raise NotADirectoryError()

    if not os.path.exists(IMAGES_DIR):
        os.makedirs(IMAGES_DIR)

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
    metadata_selected = metadata.loc[wilds_mask]
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
