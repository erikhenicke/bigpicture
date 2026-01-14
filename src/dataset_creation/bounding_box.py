"""
bounding_box.py

This module provides a bounding box class that holds four coordinates forming a rectangle.
"""

from typing import Self
import re
from dataclasses import dataclass
import warnings


@dataclass(frozen=True)
class BoundingBox:
    """Bounding box describes a geographical regions with four coordinates forming a rectangle."""
    north_west: tuple[float, float]
    north_east: tuple[float, float]
    south_east: tuple[float, float]
    south_west: tuple[float, float]

    @classmethod
    def from_raw(cls, raw_location: str) -> Self:
        """Extract coordinates from raw location string and check for validity.

        A raw location string is considered valid, if the order of points in the raw string is:
            1. North west
            2. North east
            3. South east
            4. South west

        and any additional points after the first four are duplicates.
        """
        points = re.findall(r"[-\d.]+ [-\d.]+", raw_location)
        coords: list[tuple[float, float]] = [
            (float(p.split()[0]), float(p.split()[1])) for p in points
        ]
        if len(coords) < 4:
            raise ValueError(
                f"Expected at least 4 coordinate pairs, got {len(coords)}")

        # Check if any additional points are duplicates.
        first_four: list[tuple[float, float]] = coords[:4]
        extras = coords[4:]
        if extras:
            for coord in extras:
                if coord not in first_four:
                    raise ValueError(
                        f"Extra coordinate {coord} is not one of the first four; "
                        "inconsistent bounding box"
                    )

        if contains_pole(first_four):
            raise ValueError(
                "Bounding box contains or intersects north/south pole "
                "(each corner in different longitude sector)"
            )

        if crosses_180th_meridian(first_four):
            warnings.warn('Found bounding box crossing the 180th meridian!')
            # Shift bounding box to 0th meridian
            first_four = shift_bounding_box_lon(first_four, 180)

        # Create the expected canonical form from the first four points.
        lons = [c[0] for c in first_four]
        lats = [c[1] for c in first_four]
        min_lon, max_lon = min(lons), max(lons)
        min_lat, max_lat = min(lats), max(lats)

        canonical = [
            (min_lon, max_lat),    # NW
            (max_lon, max_lat),    # NE
            (max_lon, min_lat),    # SE
            (min_lon, min_lat),    # SW
        ]

        if set(first_four) != set(canonical):
            raise ValueError(
                "First four points do not form an axis-aligned rectangle in "
                "north_west, north_east, south_east, south_west configuration"
            )

        return cls(
            north_west=canonical[0],
            north_east=canonical[1],
            south_east=canonical[2],
            south_west=canonical[3]
        )

    def as_list(self) -> list[tuple[float, float]]:
        """Return the four points in north_west, north_east, south_east, south_west order."""
        return [self.north_west, self.north_east, self.south_east, self.south_west]

    def __iter__(self):
        return iter(self.as_list())

    def get_width_deg(self) -> float:
        return abs(self.north_east[0] - self.north_west[0])

    def get_height_deg(self) -> float:
        return abs(self.north_west[1] - self.south_west[1])


def lies_any_lon_within(coords: list[tuple[float, float]], lower: float, upper: float) -> bool:
    """Checks if any longitude of the given coordinates (lon, lat) lies between lower and upper.

    Args:
        coords (list): List of coordinates (lon, lat) tuples.
        lower (float): Lower bound of the longitude sector to check.
        upper (float): Upper bound of the longitude sector to check.

    Returns:
        bool: True, if any coordinates longitude lies within the range. 
    """
    return any(lower <= coord[0] <= upper for coord in coords)


def contains_pole(bounding_box: list[tuple[float, float]]) -> bool:
    """Checks if the given bounding box contains or intersects the north or south pole.

    Args:
        bounding_box: List of four coordinate tuples (lon, lat) describing a rectangle.

    Returns:
        bool: True, if north or south pole is contained in the box, 
              i.e. each corner lies in a different longitude sector.
    """
    return (
        lies_any_lon_within(bounding_box, -180, -90)
        and lies_any_lon_within(bounding_box, -90, 0)
        and lies_any_lon_within(bounding_box, 0, 90)
        and lies_any_lon_within(bounding_box, 90, 180)
    )


def crosses_180th_meridian(bounding_box: list[tuple[float, float]]) -> bool:
    """Checks if the bounding box crosses the 180th meridian, but is canonical besides that.

    Assumptions taken: 
        - Bounding box does not span more than 20 deg in longitude.

    Args:
        bounding_box: List of four coordinate tuples (lon, lat) describing a rectangle.

    Returns:
        bool: True, if the bounding box crosses the 180th meridian and is canonical. 
    """

    c1, c2, c3, c4 = bounding_box

    canonical_cond = (
        (c1[0] == c4[0])
        and (c2[0] == c3[0])
        and (c1[1] == c2[1])
        and (c3[1] == c4[1])
        and (c1[1] > c4[1])
    )
    crossing_cond = (
        (c1[0] > 170 and c4[0] > 170)
        and (c2[0] < -170 and c3[0] < -170)
    )
    return canonical_cond and crossing_cond


def shift_bounding_box_lon(bounding_box: list[tuple[float, float]], s: float) -> list[tuple[float, float]]:
    """Shift bounding box by s degree longitude.

    Args:
        bounding_box: List of four coordinate tuples (lon, lat) describing a rectangle.

    Returns:
        Shifted bounding box. 
    """
    c1, c2, c3, c4 = bounding_box

    # Shift bounding box by s degree longitude.
    c1_shifted = (c1[0] - s, c1[1])
    c4_shifted = (c4[0] - s, c4[1])
    c2_shifted = (c2[0] + s, c2[1])
    c3_shifted = (c3[0] + s, c3[1])

    return [c1_shifted, c2_shifted, c3_shifted, c4_shifted]
