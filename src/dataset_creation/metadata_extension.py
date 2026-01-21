"""
metadata_extension.py

This module extends the WILDS metadata with image center coordinates and image span 
information in degree.
"""
import json
import os
import warnings
import pathlib

from bounding_box import BoundingBox
import pandas as pd
from wilds import get_dataset
from tqdm import tqdm
from geopy.distance import geodesic
from geopy.point import Point

tqdm.pandas()

PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"
DEST_DIR = PROJECT_ROOT.parent.parent.parent / "datasets4" / "FMoW_LandSat" / "fmow_landsat"
METADATA_DIR = DATA_DIR / "groundtruth"

if not (os.path.exists(PROJECT_ROOT) and os.path.exists(DATA_DIR) and os.path.exists(METADATA_DIR) and os.path.exists(DEST_DIR)):
    raise NotADirectoryError()


def build_fmow_metadata_path(wilds_metadata_sample: pd.core.series.Series) -> str:
    """Build sample specific path to original fmow metadata. 

    The original fmow metadata is organized as a collection of json files. To access
    the original fmow metadata, extract the split and the image filename from wilds metadata, which are of the form: 
        `<category>_<unique-sample-idx>_<sample-idx>_rgb.jpg`
    and a corresponding fmow metadata path of the form:
        `<split>/<category>/<category>_<unique-sample-idx>/tunnel_opening_370_3_rgb.json`

    Args:
        wilds_metadata_sample (pd.core.series.Series): Metadata for a sample of the fmow WILDS dataset. 

    Returns:
        str: Path to the original fmow metadata file.
    """
    split = wilds_metadata_sample["split"]
    if split == "test":
        split = "test_gt"
    img_filename = wilds_metadata_sample["img_filename"]
    img_filename_base = img_filename.split('.')[0]
    img_info = img_filename_base.split('_')

    if len(img_info) < 4:
        raise ValueError(f"Image file name {img_filename} is malformed.")

    unique_sample_idx = img_info[-3]
    category = "_".join(img_info[:-3])

    return METADATA_DIR / split / category / f"{category}_{unique_sample_idx}" / f"{img_filename_base}.json"


def compute_img_span_km(img_center_lon: float, img_center_lat: float, img_span_lat: float) -> float:
    """Compute the distance in kilometer of the image span in latitude direction.

    Args:
        img_center_lon (float): Longitude of the image center.
        img_center_lat (float): Latitude of the image center. 
        img_span_lat (float): Span of the image (north to south) in degrees latitude.  

    Returns:
        float: Span of the image in kilometer. 
    """
    # Offset is half the image span. Turn offset negative on the southern hemisphere
    lat_offset = (img_span_lat / 2) if img_center_lat > 0 else - \
        (img_span_lat / 2)
    img_nothern = Point(img_center_lat + lat_offset, img_center_lon)
    img_southern = Point(img_center_lat - lat_offset, img_center_lon)
    return geodesic(img_nothern, img_southern).km


def compute_center_coordinates_and_span(fmow_metadata_sample: pd.core.series.Series) -> tuple[tuple[float, float], float]:
    """Compute the coordinates of the image center and the image span in kilometer.

    The coordinates of the bounding box are extracted together with the image width and height.
    Both are used to interpolate the image center coordinates as well as the image span in kilometer.

    Args:
        fmow_metadata_sample (pd.core.series.Series): Metadata of the original fmow dataset. 

    Returns:
        list: Center coordinates and image span in kilometer. 
    """
    box_info = fmow_metadata_sample.get('bounding_boxes')[0]
    box = box_info.get('box')
    box_pos_x, box_pos_y, box_width, box_height = box

    img_width = fmow_metadata_sample.get('img_width')
    img_height = fmow_metadata_sample.get('img_height')
    center = (img_width / 2, img_height / 2)
    top_left_to_center_fraction = (
        (center[0] - box_pos_x) / box_width, (center[1] - box_pos_y) / box_height)

    raw_location = fmow_metadata_sample['bounding_boxes'][0]['raw_location']

    bbox = BoundingBox.from_raw(raw_location)

    center_lon = bbox.north_west[0] + \
        top_left_to_center_fraction[0] * bbox.get_width_deg()
    center_lat = bbox.north_west[1] - \
        top_left_to_center_fraction[1] * bbox.get_height_deg()

    if img_width > img_height:
        img_span_lat = img_height / box_height * bbox.get_height_deg()
    else:
        img_span_lat = img_width / box_height * bbox.get_height_deg()

    if img_span_lat > 0.1:
        warnings.warn(
            f"Very large image span of {img_span_lat} found. Probably miscalculation at {fmow_metadata_sample.get('img_filename')}!"
        )

    img_span_km = compute_img_span_km(center_lon, center_lat, img_span_lat)

    return (center_lon, center_lat), img_span_km


def extract_center_coords_and_img_span(wilds_metadata_sample):

    if wilds_metadata_sample["split"] == "seq":
        return pd.Series(
            {"img_center_lon": None,
             "img_center_lat": None,
             "img_span_km": None}
        )

    with open(build_fmow_metadata_path(wilds_metadata_sample), 'r') as file:
        fmow_metadata_sample = json.load(file)

    (center_lon, center_lat), img_span_km = compute_center_coordinates_and_span(
        fmow_metadata_sample)

    return pd.Series(
        {"img_center_lon": center_lon,
         "img_center_lat": center_lat,
         "img_span_km": img_span_km}
    )


def main():
    dataset = get_dataset(dataset="fmow")
    metadata = dataset.metadata
    coords_and_span = metadata.progress_apply(
        extract_center_coords_and_img_span, axis=1)
    metadata_extended = pd.concat([metadata, coords_and_span], axis=1)
    metadata_extended.to_csv(
        DEST_DIR / "rgb_metadata_extended.csv", index=False)


if __name__ == "__main__":
    main()
