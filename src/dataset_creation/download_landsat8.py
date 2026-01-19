"""
download_landsat8.py

This script downloads landsat8 satellite imagery for each fmow sample (WILDS version, [train, val, test]).
"""
import json
import os
import warnings
import pathlib
from functools import partial
import time
import requests
import shutil

import numpy as np
import pandas as pd
from pandarallel import pandarallel
from wilds import get_dataset
from tqdm import tqdm
import ee
from geopy.distance import geodesic
from geopy.point import Point

tqdm.pandas()
pandarallel.initialize()

PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent.resolve()
DATA_DIR = PROJECT_ROOT / "data" / "fmow_landsat"
IMAGES_DIR = DATA_DIR / "images"
EE_PROJECT_NAME = 'seeing-the-big-picture'
EXTENSION_FACTOR = 6.0
SCALE = 30.0
MAX_PIXELS = 1e8


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


def mask_l8_clouds(image: ee.Image) -> ee.Image:
    """Updates an image mask, to filter out cloudy pixels.

    For a detailed description of the 'QA_PIXEL' flags see:
        https://www.usgs.gov/landsat-missions/landsat-collection-2-quality-assessment-bands
    """
    qa = image.select('QA_PIXEL')
    # Only pixels for which the first 5 bits equal zero are not masked away.
    cloud_mask = qa.bitwiseAnd(int('11111', 2)).eq(0)
    return image.updateMask(cloud_mask)


def compute_validity_fraction(image: ee.Image, region_of_interest: ee.Geometry.Rectangle) -> ee.Number:
    """Computes the fraction of valid pixels of an image.

    Valid pixels are those, which are not masked away, i.e. whose
    mask value is equal to one.

    Args:
        image (ee.Image): Image to compute the validity for.
    Returns:
        ee.Number: Fraction of pixels, which are not masked.
    """
    return ee.Number(
        image.select(['SR_B4', 'SR_B3', 'SR_B2'])
        .mask()
        .reduce(ee.Reducer.min())
        .reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region_of_interest,
            scale=SCALE,
            maxPixels=MAX_PIXELS)
        .get('min')
    )


def add_validity(collection: ee.ImageCollection, region_of_interest: ee.Geometry.Rectangle) -> ee.ImageCollection:
    """Stacks API call to compute the fraction of valid pixels for each image in l8."""
    return collection.map(lambda img: img.set('validity', compute_validity_fraction(img, region_of_interest)))


def get_least_cloudy_single_image(collection: ee.ImageCollection, region_of_interest: ee.Geometry.Rectangle) -> ee.Image:
    """Return least cloudy image of the collection for the region."""
    return add_validity(collection, region_of_interest).sort('validity', False).first()


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
    l8_cloud_masked = l8.map(mask_l8_clouds)
    least_cloudy = get_least_cloudy_single_image(
        l8_cloud_masked, region_of_interest)
    least_cloudy_is_ok = ee.Number(least_cloudy.get('validity')).gte(0.99)
    mosaic = l8_cloud_masked.mosaic()
    return (ee.Image(ee.Algorithms.If(least_cloudy_is_ok, least_cloudy, mosaic)), region_of_interest, least_cloudy_is_ok)


def download_image(sample_metadata: pd.Series, l8: ee.ImageCollection, span_km: float, dim: int):
    """Downloads Landsat8 image from Google Earth Engine for the image coordinates of the given sample.

    Args:
        sample_metadata (pd.Series): Metadata for fmow sample containing image coordinates and span.
        l8 (ee.ImageCollection): Landsat8 image collection to download image from.
        span_km (float): Image size in kilometer to download.
        dim (int): Image pixel dimension to download.
    """

    print("Get request")
    image_request, region, is_single_image_request = get_requests(
        sample_metadata, l8, span_km
    )
    print("Got request")
    print(type(is_single_image_request))
    # print(is_single_image_request.getInfo())

    sample_idx = sample_metadata.name
    download_path = IMAGES_DIR / f"image_{sample_idx}.png"
    image_dimensions = f"{dim}x{dim}"

    url = image_request.getThumbURL({
        'region': region,
        'dimensions': image_dimensions,
        'format': 'png'})
    print(url)

    # Handle downloading the actual pixels.
    r = requests.get(url, stream=True)
    if r.status_code != 200:
        raise r.raise_for_status()

    with open(download_path, 'wb') as out_file:
        shutil.copyfileobj(r.raw, out_file)
    print("Done: ", sample_idx)


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

     l8 = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
          .map(scale_l8))

    metadata = pd.read_csv(
        DATA_DIR / "rgb_metadata_extended.csv")

    selected_splits = {"train", "val", "test"}
    is_train_val_test = metadata["split"].isin(selected_splits)

    # Sample size of metadata_selected is 470,086
    metadata_selected = metadata.loc[is_train_val_test]

    max_span = metadata_selected["img_span_km"].max()
    print(max_span)
    download_span = max_span * EXTENSION_FACTOR
    sample = metadata_selected.loc[metadata_selected["img_span_km"] == max_span].squeeze(
    )
    region = extract_region_of_interest(
        sample, download_span)
    image_size = region.area().getInfo() / (SCALE ** 2)
    image_dimension = int(image_size ** (1/2))

    test_subset = metadata_selected.sample(n=1)

    download_l8_image = partial(
        download_image, l8=l8, span_km=download_span, dim=image_dimension)

    start = time.time()
    test_subset.apply(download_l8_image, axis=1)
    end = time.time()
    print(end - start)


if __name__ == '__main__':
    main()
