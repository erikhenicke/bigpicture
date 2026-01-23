"""
download_landsat8.py

This script downloads landsat8 satellite imagery for each fmow sample (WILDS version, [train, val, test]).
"""
import os
import pathlib
from functools import partial
import time
import shutil
import random
import logging
import typing

import requests
import numpy as np
import pandas as pd
from pandarallel import pandarallel
import ee
from geopy.distance import geodesic
from geopy.point import Point

pandarallel.initialize(nb_workers=16, progress_bar=True)

PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent.resolve()
DATA_DIR = PROJECT_ROOT.parent.parent.parent / \
    "datasets4" / "FMoW_LandSat" / "fmow_landsat"
IMAGES_DIR = DATA_DIR / "images"
EE_PROJECT_NAME = 'seeing-the-big-picture'
EXTENSION_FACTOR = 3.0
SCALE = 30.0
MAX_PIXELS = 1e8
LOG_FILE = 'download.log'


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


def get_requests(sample_metadata: pd.Series, l8: ee.ImageCollection, span_km: float) -> tuple[ee.Image, ee.Geometry.Rectangle, ee.Number]:
    """Get request for least cloudy image and request for the infromation if a composite will be returned.

    Mosaic composite is requested, if single least cloudy image contains more than 1% of invalid pixels.

    Args:
        sample_metadata (pd.Series): Metadata for fmow sample containing image coordinates and span.
        l8 (ee.ImageCollection): Landsat8 image collection to request image from.
        span_km (float): Image size in kilometer to download.

    Returns:
        tuple[ee.Image, ee.Geometry.Rectangle, ee.Number]: Request for least cloudy image, region, and bool that says if a composite will be returned.
    """
    region_of_interest = extract_region_of_interest(sample_metadata, span_km)
    l8_cloud_masked = (l8
                       .filterBounds(region_of_interest)
                       .filterMetadata('CLOUD_COVER', 'less_than', 50)
                       .map(lambda img:
                            mask_clouds_compute_validity(
                                img, region_of_interest))
                       )
    least_cloudy = l8_cloud_masked.sort('validity', False).first()
    least_cloudy_is_ok = ee.Number(least_cloudy.get('validity')).gte(0.99)
    mosaic = l8_cloud_masked.mosaic()
    optical_bands = ['SR_B4', 'SR_B3', 'SR_B2']
    vis_params = {'bands': optical_bands, 'min': 0, 'max': 0.3}
    final_image = ee.Image(ee.Algorithms.If(
        least_cloudy_is_ok, least_cloudy, mosaic)).visualize(**vis_params)
    return (final_image, region_of_interest, least_cloudy_is_ok)


def download_image(sample_metadata: pd.Series, l8: ee.ImageCollection, span_km: float, logger: logging.Logger):
    """Downloads Landsat8 image from Google Earth Engine for the image coordinates of the given sample.

    Args:
        sample_metadata (pd.Series): Metadata for fmow sample containing image coordinates and span.
        l8 (ee.ImageCollection): Landsat8 image collection to download image from.
        span_km (float): Image size in kilometer to download.
        logger (logging.Logger):
    """

    image_request, region, is_single_image_request = get_requests(
        sample_metadata, l8, span_km
    )

    sample_idx = sample_metadata.name
    download_path = IMAGES_DIR / f"rgb_image_{sample_idx}.png"

    attempts = 30
    attempts_left = attempts
    while attempts_left > 0:
        try:
            url = image_request.getDownloadUrl({
                'region': region,
                'scale': 30,
                'format': 'PNG'
            })
            response = requests.get(url, stream=True)

            if response.status_code != 200:
                raise response.raise_for_status()

            with open(download_path, 'wb') as out_file:
                shutil.copyfileobj(response.raw, out_file)

        except Exception as e:
            raise e
            logger.info(f"Attempt {attempts + 1 - attempts_left} - " + str(e))
            attempts_left -= 1
            time.sleep(1 + random.uniform(0, 1))
        else:
            break


def scale_l8(image):
    scaled_optical_bands = (image
                            .select(['SR_B4', 'SR_B3', 'SR_B2'])
                            .multiply(0.0000275)
                            .add(-0.2))
    return image.addBands(scaled_optical_bands, ['SR_B4', 'SR_B3', 'SR_B2'], True)


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

    # Setup Image collection
    l8 = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
          .map(scale_l8))

    # Read metadata
    metadata = pd.read_csv(
        DATA_DIR / "rgb_metadata_extended.csv")

    selected_splits = {"train", "val", "test"}
    is_train_val_test = metadata["split"].isin(selected_splits)

    # Sample size of metadata_selected is 470,086
    metadata_selected = metadata.loc[is_train_val_test]

    max_span = metadata_selected["img_span_km"].max()
    download_span = max_span * EXTENSION_FACTOR

    test_size = 100
    test_subset = metadata_selected.sample(n=test_size)

    download_l8_image = partial(
        download_image, l8=l8, span_km=download_span, logger=logger)

    start = time.time()
    test_subset.parallel_apply(download_l8_image, axis=1)
    end = time.time()
    logger.info("Download of %d images took %s seconds.",
                test_size, f"{end - start:.2f}")


if __name__ == '__main__':
    main()
