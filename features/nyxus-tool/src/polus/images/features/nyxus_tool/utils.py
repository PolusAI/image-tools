"""Nyxus Plugin."""
import enum
import json
import logging
import os
from dataclasses import dataclass
from dataclasses import field
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any
from typing import Callable
from typing import cast

import filepattern as fp
import typer

logger = logging.getLogger(__name__)

POLUS_TAB_EXT: str = os.environ.get("POLUS_TAB_EXT", "pandas")
NUM_WORKERS: int = int(os.environ.get("NUM_WORKERS", max(cpu_count() - 1, 1)))


class Extension(str, enum.Enum):
    """Enum of File Extension."""

    PANDAS = "pandas"
    ARROW = "arrowipc"
    PARQUET = "parquet"
    DEFAULT = POLUS_TAB_EXT


@dataclass
class NyxusConfig:
    """Configuration object for Nyxus processing."""

    inp_dir: Path
    seg_dir: Path
    out_dir: Path
    features: list[str]
    single_roi: bool = False
    kwargs: dict[str, Any] = field(default_factory=dict)
    num_workers: int = NUM_WORKERS


def validate_paths(*paths: Path) -> None:
    """Validate that the provided paths exist.

    Args:
        *paths: Paths to validate.

    Raises:
        typer.BadParameter: If a path does not exist.
    """
    for p in paths:
        if not p.exists():
            msg = f"{p} does not exist"
            raise typer.BadParameter(msg)


def validate_features(features: list[str]) -> list[str]:
    """Validate the requested features against the supported feature sets.

    Args:
        features: List of requested features or groups.

    Returns:
        List of valid features with groups marked with asterisks.

    Raises:
        typer.BadParameter: If an invalid feature is requested.
    """
    valid_features = FEATURE_GROUP.union(FEATURE_LIST)
    flat_features: list[str] = [f for feat in features for f in feat.split(",")]
    invalid: list[str] = [f for f in flat_features if f not in valid_features]
    if invalid:
        msg = f"Invalid features: {', '.join(invalid)}"
        raise typer.BadParameter(msg)
    return [f"*{f}*" if f in FEATURE_GROUP else f for f in flat_features]


def write_preview(
    int_images: fp.FilePattern,
    out_dir: Path,
    file_extension: str,
    pattern: str,
) -> None:
    """Write a preview JSON file listing input/output images.

    Args:
        int_images: FilePattern object for intensity images.
        out_dir: Output directory path.
        file_extension: Output file extension.
        pattern: File pattern used to identify images.
    """
    preview_path = out_dir / "preview.json"
    out_json: dict[str, Any] = {"filepattern": pattern, "outDir": []}
    for file in int_images():
        out_name = file[1][0].name.replace(
            "".join(file[1][0].suffixes),
            f".{file_extension}",
        )
        out_json["outDir"].append(out_name)
    with preview_path.open("w", encoding="utf-8") as jfile:
        json.dump(out_json, jfile, indent=2)


class NyxusParamError(Exception):
    """Raised when a Nyxus parameter is invalid."""

    pass


NYXUS_PARAMS = {
    "neighbor_distance": {"default": 5, "type": int, "min": 1},
    "pixels_per_micron": {"default": 1.0, "type": float, "min": 1e-9},
    "coarse_gray_depth": {"default": 64, "type": int, "min": 1},
    "n_feature_calc_threads": {"default": 4, "type": int, "min": 1},
    "use_gpu_device": {"default": -1, "type": int, "min": -1},
    "ibsi": {"default": False, "type": bool},
    "gabor_kersize": {"default": 16, "type": int, "min": 1},
    "gabor_gamma": {"default": 0.1, "type": float, "min": 0.0},
    "gabor_sig2lam": {"default": 0.8, "type": float, "min": 0.0},
    "gabor_f0": {"default": 0.1, "type": float, "min": 0.0},
    "gabor_thold": {"default": 0.025, "type": float, "min": 0.0},
    "gabor_thetas": {"default": [0, 45, 90, 135], "type": list},
    "gabor_freqs": {"default": [4, 16, 32, 64], "type": list},
    "dynamic_range": {"default": 10000, "type": int, "min": 1},
    "min_intensity": {"default": 0.0, "type": float},
    "max_intensity": {"default": 1.0, "type": float},
    "ram_limit": {"default": -1, "type": int},
    "verbose": {"default": 0, "type": int, "min": 0},
    "anisotropy_x": {"default": 1.0, "type": float, "min": 1e-9},
    "anisotropy_y": {"default": 1.0, "type": float, "min": 1e-9},
}

VALID_NYXUS_KWARGS = set(NYXUS_PARAMS.keys())


class NyxusKwargType:
    """Parses a CLI KEY=VALUE string into a (key, value) tuple."""

    name = "KEY=VALUE"

    def __call__(
        self,
        value: str,
        _param: object = None,
        _ctx: object = None,
    ) -> tuple[str, int | float | bool | str]:
        """Validate KEY=VALUE argument and cast to int, float, bool, or str."""
        if isinstance(value, tuple):
            return value

        if "=" not in value:
            msg = f"'{value}' is not a valid KEY=VALUE pair"
            raise typer.BadParameter(msg)

        key, _, raw = value.partition("=")
        key = key.strip()
        raw = raw.strip()

        if key not in VALID_NYXUS_KWARGS:
            msg = f"'{key}' is not a valid Nyxus parameter"
            raise typer.BadParameter(msg)

        # Try int or float
        for _cast in (int, float):
            try:
                return key, _cast(raw)
            except ValueError:
                continue

        # Boolean
        if raw.lower() in ("true", "false"):
            return key, raw.lower() == "true"

        # fallback string
        return key, raw


# parse kwargs
def parse_nyxus_kwargs(kwargs: dict) -> dict:
    """Validate and set defaults for Nyxus parameters."""
    parsed = {}

    # Warn about unexpected keys
    unexpected_keys = set(kwargs) - set(NYXUS_PARAMS)
    if unexpected_keys:
        logger.info(
            f"Warning: unexpected keyword argument(s): {', '.join(unexpected_keys)}",
        )

    # Loop through each parameter
    for key, rules in NYXUS_PARAMS.items():
        value = kwargs.get(key, rules["default"])

        # Convert type
        if "type" in rules and rules["type"] is not None:
            typ = cast(Callable[[Any], Any], rules["type"])
            if typ is not None:
                try:
                    value = typ(value)
                except (ValueError, TypeError) as err:
                    msg = f"Parameter '{key}' must be of type {typ.__name__}"
                    raise NyxusParamError(msg) from err

        # Check minimum
        if "min" in rules and value < rules["min"]:
            msg = f"Parameter '{key}' must be >= {rules['min']}"
            raise NyxusParamError(msg)

        parsed[key] = value

    # Optional constant
    parsed["anisotropy_z"] = 1.0

    return parsed


FEATURE_GROUP = {
    "ALL_INTENSITY",
    "ALL_MORPHOLOGY",
    "BASIC_MORPHOLOGY",
    "ALL_GLCM",
    "ALL_GLRM",
    "ALL_GLRLM",
    "ALL_GLSZM",
    "ALL_GLDM",
    "ALL_NGTDM",
    "ALL_BUT_GABOR",
    "ALL_BUT_GLCM",
    "ALL",
}


FEATURE_LIST = {
    # Intensity
    "INTEGRATED_INTENSITY",
    "MEAN",
    "MEDIAN",
    "MIN",
    "MAX",
    "RANGE",
    "COVERED_IMAGE_INTENSITY_RANGE",
    "STANDARD_DEVIATION",
    "STANDARD_DEVIATION_BIASED",
    "COV",
    "STANDARD_ERROR",
    "SKEWNESS",
    "KURTOSIS",
    "EXCESS_KURTOSIS",
    "HYPERSKEWNESS",
    "HYPERFLATNESS",
    "MEAN_ABSOLUTE_DEVIATION",
    "MEDIAN_ABSOLUTE_DEVIATION",
    "ENERGY",
    "ROOT_MEAN_SQUARED",
    "ENTROPY",
    "MODE",
    "UNIFORMITY",
    "UNIFORMITY_PIU",
    "P01",
    "P10",
    "P25",
    "P75",
    "P90",
    "P99",
    "QCOD",
    "INTERQUARTILE_RANGE",
    "ROBUST_MEAN_ABSOLUTE_DEVIATION",
    "MASS_DISPLACEMENT",
    # Morphology
    "AREA_PIXELS_COUNT",
    "AREA_UM2",
    "CENTROID_X",
    "CENTROID_Y",
    "COMPACTNESS",
    "BBOX_YMIN",
    "BBOX_XMIN",
    "BBOX_HEIGHT",
    "BBOX_WIDTH",
    "MAJOR_AXIS_LENGTH",
    "MINOR_AXIS_LENGTH",
    "ECCENTRICITY",
    "ORIENTATION",
    "ROUNDNESS",
    "EXTENT",
    "ASPECT_RATIO",
    "CONVEX_HULL_AREA",
    "SOLIDITY",
    "PERIMETER",
    "EQUIVALENT_DIAMETER",
    "EDGE_MEAN_INTENSITY",
    "EDGE_STDDEV_INTENSITY",
    "EDGE_MAX_INTENSITY",
    "EDGE_MIN_INTENSITY",
    "CIRCULARITY",
    "EROSIONS_2_VANISH",
    "EROSIONS_2_VANISH_COMPLEMENT",
    "FRACT_DIM_BOXCOUNT",
    "FRACT_DIM_PERIMETER",
    "WEIGHTED_CENTROID_X",
    "WEIGHTED_CENTROID_Y",
    "MIN_FERET_DIAMETER",
    "MAX_FERET_DIAMETER",
    "MIN_FERET_ANGLE",
    "MAX_FERET_ANGLE",
    "STAT_FERET_DIAM_MIN",
    "STAT_FERET_DIAM_MAX",
    "STAT_FERET_DIAM_MEAN",
    "STAT_FERET_DIAM_MEDIAN",
    "STAT_FERET_DIAM_STDDEV",
    "STAT_FERET_DIAM_MODE",
    "STAT_MARTIN_DIAM_MIN",
    "STAT_MARTIN_DIAM_MAX",
    "STAT_MARTIN_DIAM_MEAN",
    "STAT_MARTIN_DIAM_MEDIAN",
    "STAT_MARTIN_DIAM_STDDEV",
    "STAT_MARTIN_DIAM_MODE",
    "STAT_NASSENSTEIN_DIAM_MIN",
    "STAT_NASSENSTEIN_DIAM_MAX",
    "STAT_NASSENSTEIN_DIAM_MEAN",
    "STAT_NASSENSTEIN_DIAM_MEDIAN",
    "STAT_NASSENSTEIN_DIAM_STDDEV",
    "STAT_NASSENSTEIN_DIAM_MODE",
    "MAXCHORDS_MAX",
    "MAXCHORDS_MAX_ANG",
    "MAXCHORDS_MIN",
    "MAXCHORDS_MIN_ANG",
    "MAXCHORDS_MEDIAN",
    "MAXCHORDS_MEAN",
    "MAXCHORDS_MODE",
    "MAXCHORDS_STDDEV",
    "ALLCHORDS_MAX",
    "ALLCHORDS_MAX_ANG",
    "ALLCHORDS_MIN",
    "ALLCHORDS_MIN_ANG",
    "ALLCHORDS_MEDIAN",
    "ALLCHORDS_MEAN",
    "ALLCHORDS_MODE",
    "ALLCHORDS_STDDEV",
    "EULER_NUMBER",
    "EXTREMA_P1_X"
    "EXTREMA_P1_Y"
    "EXTREMA_P2_X"
    "EXTREMA_P2_Y"
    "EXTREMA_P3_X"
    "EXTREMA_P3_Y"
    "EXTREMA_P4_X"
    "EXTREMA_P4_Y"
    "EXTREMA_P5_X"
    "EXTREMA_P5_Y"
    "EXTREMA_P6_X"
    "EXTREMA_P6_Y"
    "EXTREMA_P7_X"
    "EXTREMA_P7_Y"
    "EXTREMA_P8_X"
    "EXTREMA_P8_Y"
    "POLYGONALITY_AVE",
    "HEXAGONALITY_AVE",
    "HEXAGONALITY_STDDEV",
    "DIAMETER_MIN_ENCLOSING_CIRCLE",
    "DIAMETER_CIRCUMSCRIBING_CIRCLE",
    "DIAMETER_INSCRIBING_CIRCLE",
    "GEODETIC_LENGTH",
    "THICKNESS",
    "ROI_RADIUS_MEAN",
    "ROI_RADIUS_MAX",
    "ROI_RADIUS_MEDIAN",
    # GLCM texture
    "GLCM_ASM",
    "GLCM_ACOR",
    "GLCM_CLUPROM",
    "GLCM_CLUSHADE",
    "GLCM_CLUTEND",
    "GLCM_CONTRAST",
    "GLCM_CORRELATION",
    "GLCM_DIFAVE",
    "GLCM_DIFENTRO",
    "GLCM_DIFVAR",
    "GLCM_DIS",
    "GLCM_ENERGY",
    "GLCM_ENTROPY",
    "GLCM_HOM1",
    "GLCM_HOM2",
    "GLCM_ID",
    "GLCM_IDN",
    "GLCM_IDM",
    "GLCM_IDMN",
    "GLCM_INFOMEAS1",
    "GLCM_INFOMEAS2",
    "GLCM_IV",
    "GLCM_JAVE",
    "GLCM_JE",
    "GLCM_JMAX",
    "GLCM_JVAR",
    "GLCM_SUMAVERAGE",
    "GLCM_SUMENTROPY",
    "GLCM_SUMVARIANCE",
    "GLCM_VARIANCE",
    # GLRLM texture
    "GLRLM_SRE",
    "GLRLM_LRE",
    "GLRLM_GLN",
    "GLRLM_GLNN",
    "GLRLM_RLN",
    "GLRLM_RLNN",
    "GLRLM_RP",
    "GLRLM_GLV",
    "GLRLM_RV",
    "GLRLM_RE",
    "GLRLM_LGLRE",
    "GLRLM_HGLRE",
    "GLRLM_SRLGLE",
    "GLRLM_SRHGLE",
    "GLRLM_LRLGLE",
    "GLRLM_LRHGLE",
    # GLDZM texture
    "GLDZM_SDE",
    "GLDZM_LDE",
    "GLDZM_LGLE",
    "GLDZM_HGLE",
    "GLDZM_SDLGLE",
    "GLDZM_SDHGLE",
    "GLDZM_LDLGLE",
    "GLDZM_LDHGLE",
    "GLDZM_GLNU",
    "GLDZM_GLNUN",
    "GLDZM_ZDNU",
    "GLDZM_ZDNUN",
    "GLDZM_ZP",
    "GLDZM_GLM",
    "GLDZM_GLV",
    "GLDZM_ZDM",
    "GLDZM_ZDV",
    "GLDZM_ZDE",
    # GLSZM texture
    "GLSZM_SAE",
    "GLSZM_LAE",
    "GLSZM_GLN",
    "GLSZM_GLNN",
    "GLSZM_SZN",
    "GLSZM_SZNN",
    "GLSZM_ZP",
    "GLSZM_GLV",
    "GLSZM_ZV",
    "GLSZM_ZE",
    "GLSZM_LGLZE",
    "GLSZM_HGLZE",
    "GLSZM_SALGLE",
    "GLSZM_SAHGLE",
    "GLSZM_LALGLE",
    "GLSZM_LAHGLE",
    "GLDM_SDE",
    "GLDM_LDE",
    "GLDM_GLN",
    "GLDM_DN",
    "GLDM_DNN",
    "GLDM_GLV",
    "GLDM_DV",
    "GLDM_DE",
    "GLDM_LGLE",
    "GLDM_HGLE",
    "GLDM_SDLGLE",
    "GLDM_SDHGLE",
    "GLDM_LDLGLE",
    "GLDM_LDHGLE",
    "NGLDM_LDE",
    "NGLDM_HDE",
    "NGLDM_LGLCE",
    "NGLDM_HGLCE",
    "NGLDM_LDLGLE",
    "NGLDM_LDHGLE",
    "NGLDM_HDLGLE",
    "NGLDM_HDHGLE",
    "NGLDM_GLNU",
    "NGLDM_GLNUN",
    "NGLDM_DCNU",
    "NGLDM_DCNUN",
    "NGLDM_GLM",
    "NGLDM_GLV",
    "NGLDM_DCM",
    "NGLDM_DCV",
    "NGLDM_DCE",
    "NGLDM_DCENE",
    "NGTDM_COARSENESS",
    "NGTDM_CONTRAST",
    "NGTDM_BUSYNESS",
    "NGTDM_COMPLEXITY",
    "NGTDM_STRENGTH",
    # Radial / frequency
    "ZERNIKE2D",
    "FRAC_AT_D",
    "MEAN_FRAC",
    "RADIAL_CV",
    "GABOR",
    # Image moments (example expansion up to order 3)
    "SPAT_MOMENT_00",
    "SPAT_MOMENT_01",
    "SPAT_MOMENT_02",
    "SPAT_MOMENT_03",
    "SPAT_MOMENT_10",
    "SPAT_MOMENT_11",
    "SPAT_MOMENT_12",
    "SPAT_MOMENT_20",
    "SPAT_MOMENT_21",
    "SPAT_MOMENT_30",
    "WEIGHTED_SPAT_MOMENT_00",
    "WEIGHTED_SPAT_MOMENT_01",
    "WEIGHTED_SPAT_MOMENT_02",
    "WEIGHTED_SPAT_MOMENT_03",
    "WEIGHTED_SPAT_MOMENT_10",
    "WEIGHTED_SPAT_MOMENT_11",
    "WEIGHTED_SPAT_MOMENT_20",
    "WEIGHTED_SPAT_MOMENT_21",
    "WEIGHTED_SPAT_MOMENT_30",
    "CENTRAL_MOMENT_00",
    "CENTRAL_MOMENT_01",
    "CENTRAL_MOMENT_02",
    "CENTRAL_MOMENT_03",
    "CENTRAL_MOMENT_10",
    "CENTRAL_MOMENT_11",
    "CENTRAL_MOMENT_12",
    "CENTRAL_MOMENT_20",
    "CENTRAL_MOMENT_21",
    "CENTRAL_MOMENT_30",
    "WEIGHTED_CENTRAL_MOMENT_02",
    "WEIGHTED_CENTRAL_MOMENT_03",
    "WEIGHTED_CENTRAL_MOMENT_11",
    "WEIGHTED_CENTRAL_MOMENT_12",
    "WEIGHTED_CENTRAL_MOMENT_20",
    "WEIGHTED_CENTRAL_MOMENT_21",
    "WEIGHTED_CENTRAL_MOMENT_30",
    "NORM_CENTRAL_MOMENT_02",
    "NORM_CENTRAL_MOMENT_03",
    "NORM_CENTRAL_MOMENT_11",
    "NORM_CENTRAL_MOMENT_12",
    "NORM_CENTRAL_MOMENT_20",
    "NORM_CENTRAL_MOMENT_21",
    "NORM_CENTRAL_MOMENT_30",
    "NORM_SPAT_MOMENT_00",
    "NORM_SPAT_MOMENT_01",
    "NORM_SPAT_MOMENT_02",
    "NORM_SPAT_MOMENT_03",
    "NORM_SPAT_MOMENT_10",
    "NORM_SPAT_MOMENT_20",
    "NORM_SPAT_MOMENT_30",
    # Hu moments
    "HU_M1",
    "HU_M2",
    "HU_M3",
    "HU_M4",
    "HU_M5",
    "HU_M6",
    "HU_M7",
    "WEIGHTED_HU_M1",
    "WEIGHTED_HU_M2",
    "WEIGHTED_HU_M3",
    "WEIGHTED_HU_M4",
    "WEIGHTED_HU_M5",
    "WEIGHTED_HU_M6",
    "WEIGHTED_HU_M7",
    # IMOM features
    "IMOM_RM_00",
    "IMOM_RM_01",
    "IMOM_RM_02",
    "IMOM_RM_03",
    "IMOM_RM_10",
    "IMOM_RM_11",
    "IMOM_RM_12",
    "IMOM_RM_20",
    "IMOM_RM_21",
    "IMOM_RM_30",
    "IMOM_WRM_00",
    "IMOM_WRM_01",
    "IMOM_WRM_02",
    "IMOM_WRM_03",
    "IMOM_WRM_10",
    "IMOM_WRM_11",
    "IMOM_WRM_12",
    "IMOM_WRM_20",
    "IMOM_WRM_21",
    "IMOM_WRM_30",
    "IMOM_CM_02",
    "IMOM_CM_03",
    "IMOM_CM_11",
    "IMOM_CM_12",
    "IMOM_CM_20",
    "IMOM_CM_21",
    "IMOM_CM_30",
    "IMOM_WCM_02",
    "IMOM_WCM_03",
    "IMOM_WCM_11",
    "IMOM_WCM_12",
    "IMOM_WCM_20",
    "IMOM_WCM_21",
    "IMOM_WCM_30",
    "IMOM_NCM_02",
    "IMOM_NCM_03",
    "IMOM_NCM_11",
    "IMOM_NCM_20",
    "IMOM_NCM_21",
    "IMOM_NCM_30",
    "IMOM_NRM_00",
    "IMOM_NRM_01",
    "IMOM_NRM_02",
    "IMOM_NRM_03",
    "IMOM_NRM_10",
    "IMOM_NRM_20",
    "IMOM_NRM_30",
    "IMOM_HU1",
    "IMOM_HU2",
    "IMOM_HU3",
    "IMOM_HU4",
    "IMOM_HU5",
    "IMOM_HU6",
    "IMOM_HU7",
    "IMOM_WHU1",
    "IMOM_WHU2",
    "IMOM_WHU3",
    "IMOM_WHU4",
    "IMOM_WHU5",
    "IMOM_WHU6",
    "IMOM_WHU7",
    # Neighbor
    "NUM_NEIGHBORS",
    "PERCENT_TOUCHING",
    "CLOSEST_NEIGHBOR1_DIST",
    "CLOSEST_NEIGHBOR1_ANG",
    "CLOSEST_NEIGHBOR2_DIST",
    "CLOSEST_NEIGHBOR2_ANG",
    "ANG_BW_NEIGHBORS_MEAN",
    "ANG_BW_NEIGHBORS_STDDEV",
    "ANG_BW_NEIGHBORS_MODE",
}
