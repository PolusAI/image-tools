"""Ome micojson package."""

import json
import logging
import os
from pathlib import Path

import microjson.model as mj
from microjson.tilemodel import TileJSON
from microjson.tilemodel import TileLayer
from microjson.tilemodel import TileModel
from microjson.tilewriter import TileWriter
from microjson.tilewriter import extract_fields_ranges_enums
from microjson.tilewriter import getbounds

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


FEATURE_GROUP = {
    "ALL_INTENSITY",
    "ALL_MORPHOLOGY",
    "BASIC_MORPHOLOGY",
    "ALL_GLCM",
    "ALL_GLRLM",
    "ALL_GLSZM",
    "ALL_GLDM",
    "ALL_NGTDM",
    "ALL_EASY",
    "ALL",
}
FEATURE_LIST = {
    "INTEGRATED_INTENSITY",
    "MEAN",
    "MAX",
    "MEDIAN",
    "STANDARD_DEVIATION",
    "MODE",
    "SKEWNESS",
    "KURTOSIS",
    "HYPERSKEWNESS",
    "HYPERFLATNESS",
    "MEAN_ABSOLUTE_DEVIATION",
    "ENERGY",
    "ROOT_MEAN_SQUARED",
    "ENTROPY",
    "UNIFORMITY",
    "UNIFORMITY_PIU",
    "P01",
    "P10",
    "P25",
    "P75",
    "P90",
    "P99",
    "INTERQUARTILE_RANGE",
    "ROBUST_MEAN_ABSOLUTE_DEVIATION",
    "MASS_DISPLACEMENT",
    "AREA_PIXELS_COUNT",
    "COMPACTNESS",
    "BBOX_YMIN",
    "BBOX_XMIN",
    "BBOX_HEIGHT",
    "BBOX_WIDTH",
    "MINOR_AXIS_LENGTH",
    "MAGOR_AXIS_LENGTH",
    "ECCENTRICITY",
    "ORIENTATION",
    "ROUNDNESS",
    "NUM_NEIGHBORS",
    "PERCENT_TOUCHING",
    "EXTENT",
    "CONVEX_HULL_AREA",
    "SOLIDITY",
    "PERIMETER",
    "EQUIVALENT_DIAMETER",
    "EDGE_MEAN",
    "EDGE_MAX",
    "EDGE_MIN",
    "EDGE_STDDEV_INTENSITY",
    "CIRCULARITY",
    "EROSIONS_2_VANISH",
    "EROSIONS_2_VANISH_COMPLEMENT",
    "FRACT_DIM_BOXCOUNT",
    "FRACT_DIM_PERIMETER",
    "GLCM",
    "GLRLM",
    "GLSZM",
    "GLDM",
    "NGTDM",
    "ZERNIKE2D",
    "FRAC_AT_D",
    "RADIAL_CV",
    "MEAN_FRAC",
    "GABOR",
    "ALL_INTENSITY",
    "ALL_MORPHOLOGY",
    "BASIC_MORPHOLOGY",
    "ALL_GLCM",
    "ALL_GLRLM",
    "ALL_GLSZM",
    "ALL_GLDM",
    "ALL_NGTDM",
    "ALL_EASY",
    "ALL",
}


def convert_microjson_tile_json(microjson_path: Path) -> None:
    """Converts a MicroJSON file to TileJSON format.

    Args:
        microjson_path: Path to the input MicroJSON file.

    Outputs:
        - A `metadata.json` file with TileJSON metadata.
        - A `tiles` directory containing generated PBF tile files.
    """
    if microjson_path:
        # Extract fields, ranges, enums from the provided MicroJSON
        field_names, field_ranges, field_enums = extract_fields_ranges_enums(
            microjson_path,
        )

        # Create a TileLayer including the extracted fields
        vector_layers = [
            TileLayer(
                id="extracted-layer",
                fields=field_names,
                minzoom=0,
                maxzoom=10,
                description="Layer with extracted fields",
                fieldranges=field_ranges,
                fieldenums=field_enums,
            ),
        ]

        # # create the tiles directory
        out_dir = microjson_path.parent.joinpath(
            Path(Path(microjson_path.name).stem).joinpath("tiles"),
        )

    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)

    # get bounds
    maxbounds = getbounds(microjson_path)

    center = [
        0,
        (maxbounds[0] + maxbounds[2]) / 2,
        (maxbounds[1] + maxbounds[3]) / 2,
    ]

    # Instantiate TileModel with your settings
    tile_model = TileModel(
        tilejson="3.0.0",
        tiles=["{z}/{x}/{y}.pbf"],
        name="Example Tile Layer",
        description="A TileJSON example incorporating MicroJSON data",
        version="1.0.0",
        attribution="Polus AI",
        minzoom=0,
        maxzoom=7,
        bounds=maxbounds,
        center=center,
        vector_layers=vector_layers,
    )

    # Create the root model with your TileModel instance
    tileobj = TileJSON(root=tile_model)

    tilepath = out_dir.joinpath("metadata.json")
    with Path.open(tilepath, "w") as f:
        f.write(tileobj.model_dump_json(indent=2))

    # # Initialize the TileHandler
    handler = TileWriter(tile_model, pbf=True)
    os.chdir(tilepath.parent)
    handler.microjson2tiles(microjson_path, validate=False)


def generate_feature_collection() -> mj.MicroFeatureCollection:
    """Generate microjson feature collection."""
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [46.0, 481.7],
                            [46.7, 481.0],
                            [47.0, 480.7],
                            [48.0, 480.7],
                            [49.0, 480.7],
                            [50.0, 480.7],
                            [50.7, 480.0],
                            [51.0, 479.7],
                            [52.0, 479.7],
                            [52.7, 479.0],
                            [53.0, 478.7],
                            [54.0, 478.7],
                            [55.0, 478.7],
                            [56.0, 478.7],
                            [57.0, 478.7],
                            [57.7, 478.0],
                            [57.7, 477.0],
                            [57.7, 476.0],
                            [57.7, 475.0],
                            [57.0, 474.3],
                            [56.7, 474.0],
                            [56.7, 473.0],
                            [56.7, 472.0],
                            [56.0, 471.3],
                            [55.7, 471.0],
                            [55.0, 470.3],
                            [54.7, 470.0],
                            [54.0, 469.3],
                            [53.7, 469.0],
                            [53.7, 468.0],
                            [53.0, 467.3],
                            [52.0, 467.3],
                            [51.0, 467.3],
                            [50.0, 467.3],
                            [49.0, 467.3],
                            [48.0, 467.3],
                            [47.3, 468.0],
                            [47.0, 468.3],
                            [46.0, 468.3],
                            [45.0, 468.3],
                            [44.0, 468.3],
                            [43.3, 469.0],
                            [43.0, 469.3],
                            [42.3, 470.0],
                            [42.3, 471.0],
                            [42.0, 471.3],
                            [41.3, 472.0],
                            [41.3, 473.0],
                            [41.0, 473.3],
                            [40.3, 474.0],
                            [40.3, 475.0],
                            [40.3, 476.0],
                            [40.3, 477.0],
                            [40.3, 478.0],
                            [41.0, 478.7],
                            [41.3, 479.0],
                            [41.3, 480.0],
                            [42.0, 480.7],
                            [42.3, 481.0],
                            [43.0, 481.7],
                            [44.0, 481.7],
                            [45.0, 481.7],
                            [46.0, 481.7],
                        ],
                    ],
                },
                "properties": {
                    "Image": "p4_y1_r(289-375)_c0.ome.tif",
                    "X": 2650,
                    "Y": 2384,
                    "Label": 1,
                },
            },
        ],
    }


def preview_data(out_dir: Path) -> None:
    """Get Preview Data."""
    microjson_output = generate_feature_collection()
    out_file = out_dir.joinpath("example_data.json")

    with Path.open(out_file, "w") as json_file:
        json.dump(microjson_output, json_file, indent=2)
