"""Image registration: ORB/FLANN matching, homography, and tiled bfio read/write."""
from __future__ import annotations

import argparse
import logging
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
from pathlib import Path
from typing import cast

import cv2
import numpy as np
from bfio.bfio import BioReader
from bfio.bfio import BioWriter

CORRELATION_FALLBACK_THRESHOLD = 0.4
TILE_MERGE_THRESHOLD = 3072
TARGET_PIXELS = 5_000_000  # 5 megapixels downscale target


def corr2(a: np.ndarray, b: np.ndarray) -> float:
    """Correlation between two same-shaped 2D arrays (Pearson on flattened data).

    Args:
        a: First image or tile.
        b: Second image or tile.

    Returns:
        Scalar correlation coefficient.
    """
    c = np.sum(a) / np.size(a)
    d = np.sum(b) / np.size(b)

    c = a - c
    d = b - d

    return float((c * d).sum() / np.sqrt((c * c).sum() * (d * d).sum()))


def get_transform(
    moving_image: np.ndarray,
    reference_image: np.ndarray,
    max_val: float,
    min_val: float,
    method: str,
) -> np.ndarray | None:
    """Estimate homography or affine map from moving_image to reference_image.

    Args:
        moving_image: Image to align.
        reference_image: Fixed reference.
        max_val: Normalization upper bound.
        min_val: Normalization lower bound.
        method: One of ``Projective``, ``Affine``, or ``PartialAffine``.

    Returns:
        Transform matrix, or ``None`` if matching fails.
    """
    # max number of features to be calculated using ORB
    max_features = 500000
    # initialize orb feature matcher
    orb = cv2.ORB_create(max_features)

    # Normalize images and convert to appropriate type
    moving_image_norm = cv2.GaussianBlur(moving_image, (3, 3), 0)
    reference_image_norm = cv2.GaussianBlur(reference_image, (3, 3), 0)
    moving_image_norm = (moving_image_norm - min_val) / (max_val - min_val)
    moving_image_norm = (moving_image_norm * 255).astype(np.uint8)
    reference_image_norm = (reference_image_norm - min_val) / (max_val - min_val)
    reference_image_norm = (reference_image_norm * 255).astype(np.uint8)

    # find keypoints and descriptors in moving and reference image
    keypoints1, descriptors1 = orb.detectAndCompute(moving_image_norm, None)
    keypoints2, descriptors2 = orb.detectAndCompute(reference_image_norm, None)

    # Escape if one image does not have descriptors
    d1_ok = isinstance(descriptors1, np.ndarray)
    d2_ok = isinstance(descriptors2, np.ndarray)
    desc_ok = d1_ok and d2_ok
    if not desc_ok:
        return None

    # match and sort the descriptos using hamming distance
    flann_params = {
        "algorithm": 6,  # FLANN_INDEX_LSH
        "table_number": 6,
        "key_size": 12,
        "multi_probe_level": 1,
    }
    matcher = cv2.FlannBasedMatcher(flann_params, {})
    matches = matcher.match(descriptors1, descriptors2, None)
    matches.sort(key=lambda x: x.distance, reverse=False)

    # extract top 25% of matches
    good_match_percent = 0.25
    num_good_matches = int(len(matches) * good_match_percent)
    matches = matches[:num_good_matches]

    # extract the point coordinates from the keypoints
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)
    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # If no matching points, return None
    if points1.shape[0] == 0 or points2.shape[0] == 0:
        return None

    # calculate the homography matrix
    homography: np.ndarray
    if method == "Projective":
        homography, _ = cv2.findHomography(points1, points2, cv2.RANSAC)
    elif method == "Affine":
        homography, _ = cv2.estimateAffine2D(points1, points2, method=cv2.RANSAC)
    elif method == "PartialAffine":
        homography, _ = cv2.estimateAffinePartial2D(points1, points2, method=cv2.RANSAC)
    else:
        return None

    return homography


def get_scale_factor(height: int, width: int) -> int:
    """Scale factor to cap effective resolution near ``TARGET_PIXELS``.

    Args:
        height: Image height.
        width: Image width.

    Returns:
        Integer scale factor (at least 1).
    """
    target_size = TARGET_PIXELS
    scale_factor = np.sqrt((height * width) / target_size)
    return int(scale_factor) if scale_factor > 1 else 1


def get_scaled_down_images(
    image: BioReader,
    scale_factor: int,
    get_max: bool = False,
) -> np.ndarray | tuple[np.ndarray, float, float]:
    """Return a downscaled view of a large BioReader image (optionally min/max).

    Args:
        image: Open ``BioReader`` instance.
        scale_factor: Integer downscale factor.
        get_max: If True, also return max and min pixel values from tiles.

    Returns:
        Downscaled array, or ``(array, max_val, min_val)`` when ``get_max`` is True.
    """
    # Calculate scaling variables
    stride = int(scale_factor * np.floor(4096 / scale_factor))
    width = np.ceil(image.num_y() / scale_factor).astype(int)
    height = np.ceil(image.num_x() / scale_factor).astype(int)

    # Initialize the output
    rescaled_image = np.zeros((width, height), dtype=image._pix["type"])

    def load_and_scale(  # noqa: PLR0913
        x_read_bounds: list[int],
        y_read_bounds: list[int],
        x_write_bounds: list[int],
        y_write_bounds: list[int],
        get_max: bool = get_max,
        reader: BioReader = image,
        scale_factor: int = scale_factor,
        rescaled_image: np.ndarray = rescaled_image,
    ) -> tuple[float, float] | None:
        """Load a tile, downscale, and write into ``rescaled_image`` (thread worker)."""
        # Read an image tile
        tile = reader.read_image(
            X=x_read_bounds,
            Y=y_read_bounds,
            Z=[0, 1],
            C=[0],
            T=[0],
        ).squeeze()

        # Average the image for scaling
        blurred_image = cv2.boxFilter(tile, -1, (scale_factor, scale_factor))

        # Collect pixels for downscaled image
        y0, y1 = y_write_bounds[0], y_write_bounds[1]
        x0, x1 = x_write_bounds[0], x_write_bounds[1]
        rescaled_image[y0:y1, x0:x1] = blurred_image[::scale_factor, ::scale_factor]

        if get_max:
            return np.max(tile), np.min(tile)
        return None

    # Load and downscale the image
    threads = []
    with ThreadPoolExecutor(max([cpu_count() // 2, 1])) as executor:
        for x in range(0, image.num_x(), stride):
            x_max = np.min([x + stride, image.num_x()])  # max x to load
            xi = int(x // scale_factor)  # initial scaled x-index
            xe = int(np.ceil(x_max / scale_factor))  # ending scaled x-index
            for y in range(0, image.num_y(), stride):
                y_max = np.min([y + stride, image.num_y()])  # max y to load
                yi = int(y // scale_factor)  # initial scaled y-index
                ye = int(np.ceil(y_max / scale_factor))  # ending scaled y-index

                threads.append(
                    executor.submit(
                        load_and_scale,
                        [x, x_max],
                        [y, y_max],
                        [xi, xe],
                        [yi, ye],
                    ),
                )

    # Return max and min values if requested
    if get_max:
        tile_ranges: list[tuple[float, float]] = []
        for thread in threads:
            out = cast(tuple[float, float], thread.result())
            tile_ranges.append((float(out[0]), float(out[1])))
        max_val = max(t[0] for t in tile_ranges)
        min_val = min(t[1] for t in tile_ranges)
        return rescaled_image, max_val, min_val
    return rescaled_image


def register_image(  # noqa: PLR0913
    br_ref: BioReader,
    br_mov: BioReader,
    bw: BioWriter,
    xt_bounds: list[int],
    yt_bounds: list[int],
    xm_bounds: list[int],
    ym_bounds: list[int],
    tile_x: int,
    tile_y: int,
    x_crop_bounds: list[int],
    y_crop_bounds: list[int],
    max_val: float,
    min_val: float,
    method: str,
    rough_homography_upscaled: np.ndarray,
) -> np.ndarray:
    """Register one tile pair and write the warped moving tile to ``bw``.

    Args:
        br_ref: Reference ``BioReader``.
        br_mov: Moving ``BioReader``.
        bw: Output ``BioWriter``.
        xt_bounds: Reference X span ``[start, end)``.
        yt_bounds: Reference Y span ``[start, end)``.
        xm_bounds: Moving X span.
        ym_bounds: Moving Y span.
        tile_x: Output tile X index.
        tile_y: Output tile Y index.
        x_crop_bounds: Crop range in X after warp.
        y_crop_bounds: Crop range in Y after warp.
        max_val: Intensity normalization max.
        min_val: Intensity normalization min.
        method: Transform kind.
        rough_homography_upscaled: Fallback transform if local match fails.

    Returns:
        Homography or affine matrix used for this tile.
    """
    # Load a section of the reference and moving images
    ref_tile = br_ref.read_image(
        X=[xt_bounds[0], xt_bounds[1]],
        Y=[yt_bounds[0], yt_bounds[1]],
        Z=[0, 1],
        C=[0],
        T=[0],
    ).squeeze()
    mov_tile = br_mov.read_image(
        X=[xm_bounds[0], xm_bounds[1]],
        Y=[ym_bounds[0], ym_bounds[1]],
        Z=[0, 1],
        C=[0],
        T=[0],
    ).squeeze()

    # Get the transformation matrix
    projective_transform = get_transform(mov_tile, ref_tile, max_val, min_val, method)

    # Use the rough transformation matrix if no matrix was returned
    is_rough = False
    if not isinstance(projective_transform, np.ndarray):
        is_rough = True
        projective_transform = rough_homography_upscaled

    # Transform the moving image
    if method == "Projective":
        transformed_image = cv2.warpPerspective(
            mov_tile,
            projective_transform,
            (xt_bounds[1] - xt_bounds[0], yt_bounds[1] - yt_bounds[0]),
        )
    else:
        transformed_image = cv2.warpAffine(
            mov_tile,
            projective_transform,
            (xt_bounds[1] - xt_bounds[0], yt_bounds[1] - yt_bounds[0]),
        )

    # Determine the correlation between the reference and transformed moving image
    corr = corr2(ref_tile, transformed_image)

    # If the correlation is bad, try using the rough transform instead
    if corr < CORRELATION_FALLBACK_THRESHOLD and not is_rough:
        if method == "Projective":
            transformed_image = cv2.warpPerspective(
                mov_tile,
                rough_homography_upscaled,
                (xt_bounds[1] - xt_bounds[0], yt_bounds[1] - yt_bounds[0]),
            )
        else:
            transformed_image = cv2.warpAffine(
                mov_tile,
                rough_homography_upscaled,
                (xt_bounds[1] - xt_bounds[0], yt_bounds[1] - yt_bounds[0]),
            )
        projective_transform = rough_homography_upscaled

    # Write the transformed moving image
    bw.write_image(
        transformed_image[
            y_crop_bounds[0] : y_crop_bounds[1],
            x_crop_bounds[0] : x_crop_bounds[1],
            np.newaxis,
            np.newaxis,
            np.newaxis,
        ],
        X=[tile_x],
        Y=[tile_y],
    )

    return projective_transform


def apply_transform(  # noqa: PLR0913
    br_mov: BioReader,
    bw: BioWriter,
    tiles: tuple[list[int], list[int], list[int], list[int]],
    shape: tuple[int, int, list[int], list[int]],
    transform: np.ndarray,
    method: str,
) -> None:
    """Apply an existing transform to one moving tile and write to ``bw``.

    Args:
        br_mov: Moving ``BioReader``.
        bw: Output ``BioWriter``.
        tiles: ``(xm_bounds, ym_bounds, xt_bounds, yt_bounds)``.
        shape: ``(tile_x, tile_y, x_crop_bounds, y_crop_bounds)``.
        transform: Matrix from ``register_image``.
        method: ``Projective`` or affine family.
    """
    xm_bounds, ym_bounds, xt_bounds, yt_bounds = tiles

    # Read the moving image tile
    mov_tile = br_mov.read_image(
        X=[xm_bounds[0], xm_bounds[1]],
        Y=[ym_bounds[0], ym_bounds[1]],
        Z=[0, 1],
        C=[0],
        T=[0],
    ).squeeze()

    # Get the image coordinates and shape
    tile_x, tile_y, x_crop_bounds, y_crop_bounds = shape

    # Transform the moving image
    if method == "Projective":
        transformed_image = cv2.warpPerspective(
            mov_tile,
            transform,
            (xt_bounds[1] - xt_bounds[0], yt_bounds[1] - yt_bounds[0]),
        )
    else:
        transformed_image = cv2.warpAffine(
            mov_tile,
            transform,
            (xt_bounds[1] - xt_bounds[0], yt_bounds[1] - yt_bounds[0]),
        )

    # Write the transformed image to the output file
    bw.write_image(
        transformed_image[
            y_crop_bounds[0] : y_crop_bounds[1],
            x_crop_bounds[0] : x_crop_bounds[1],
            np.newaxis,
            np.newaxis,
            np.newaxis,
        ],
        X=[tile_x],
        Y=[tile_y],
    )


if __name__ == "__main__":
    # Initialize the logger
    logging.basicConfig(
        format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
        datefmt="%d-%b-%y %H:%M:%S",
    )
    logger = logging.getLogger("image_registration.py")
    logger.setLevel(logging.INFO)

    # Setup the argument parsing
    logger.info("Parsing arguments...")
    parser = argparse.ArgumentParser(
        prog="imageRegistration",
        description="This script registers an image collection",
    )
    parser.add_argument(
        "--registrationString",
        dest="registration_string",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--similarTransformationString",
        dest="similar_transformation_string",
        type=str,
        required=True,
    )
    parser.add_argument("--outDir", dest="outDir", type=str, required=True)
    parser.add_argument("--template", dest="template", type=str, required=True)
    parser.add_argument("--method", dest="method", type=str, required=True)

    # parse the arguments
    args = parser.parse_args()
    registration_string = args.registration_string
    similar_transformation_string = args.similar_transformation_string
    output_dir = args.outDir
    template = args.template
    method = args.method

    # Set up the number of threads for each task
    read_workers = max([cpu_count() // 3, 1])
    write_workers = max([cpu_count() - 1, 2])
    loop_workers = max([3 * cpu_count() // 4, 2])

    # extract filenames from registration_string and similar_transformation_string
    registration_set = registration_string.split()
    similar_transformation_set = similar_transformation_string.split()

    filename_len = len(template)

    # read and downscale reference image
    br_ref = BioReader(registration_set[0], max_workers=write_workers)
    scale_factor = get_scale_factor(br_ref.num_y(), br_ref.num_x())
    logger.info(f"Scale factor: {scale_factor}")

    # scale matrix used to upscale transformation matrices
    if method == "Projective":
        inv_sf = 1 / scale_factor
        scale_matrix = np.array(
            [
                [1, 1, scale_factor],
                [1, 1, scale_factor],
                [inv_sf, inv_sf, 1],
            ],
        )
    else:
        inv_sf = 1 / scale_factor
        row = [inv_sf, inv_sf, 1]
        scale_matrix = np.array([row, row])

    ref_name = Path(registration_set[0]).name
    logger.info(f"Reading and downscaling reference image: {ref_name}")
    ref_ds, max_val, min_val = get_scaled_down_images(
        br_ref,
        scale_factor,
        get_max=True,
    )
    br_ref.max_workers = read_workers

    # read moving image
    mov_name = Path(registration_set[1]).name
    logger.info(f"Reading and downscaling moving image: {mov_name}")
    br_mov = BioReader(registration_set[1], max_workers=write_workers)
    moving_ds = get_scaled_down_images(br_mov, scale_factor)
    br_mov.max_workers = read_workers

    # calculate rough transformation between scaled down reference and moving image
    logger.info("calculating rough homography...")
    rough_homography_downscaled = get_transform(
        moving_ds,
        ref_ds,
        max_val,
        min_val,
        method,
    )

    # upscale the rough homography matrix
    logger.info("Inverting homography...")
    if method == "Projective":
        rough_homography_upscaled = rough_homography_downscaled * scale_matrix
        homography_inverse = np.linalg.inv(rough_homography_upscaled)
    else:
        rough_homography_upscaled = rough_homography_downscaled
        homography_inverse = cv2.invertAffineTransform(rough_homography_downscaled)

    # Initialize the output file
    out_path = str(Path(output_dir).joinpath(Path(registration_set[1]).name))
    bw = BioWriter(
        out_path,
        metadata=br_mov.read_metadata(),
        max_workers=write_workers,
    )
    bw.num_x(br_ref.num_x())
    bw.num_y(br_ref.num_y())
    bw.num_z(1)
    bw.num_c(1)
    bw.num_t(1)

    # transformation variables
    reg_shape = []
    reg_tiles = []
    reg_homography = []

    # Loop through image tiles and start threads
    logger.info("Starting threads...")
    threads = []
    first_tile = True
    with ThreadPoolExecutor(max_workers=loop_workers) as executor:
        for x in range(0, br_ref.num_x(), 2048):
            for y in range(0, br_ref.num_y(), 2048):
                # Get reference/template image coordinates
                nx, ny = br_ref.num_x(), br_ref.num_y()
                xt_hi = min(nx, x + 2048 + 1024)
                yt_hi = min(ny, y + 2048 + 1024)
                xt_bounds = [max(0, x - 1024), xt_hi]
                yt_bounds = [max(0, y - 1024), yt_hi]

                # Use the rough homography to get coordinates in the moving image
                coords = np.array(
                    [
                        [xt_bounds[0], xt_bounds[0], xt_bounds[1], xt_bounds[1]],
                        [yt_bounds[0], yt_bounds[1], yt_bounds[1], yt_bounds[0]],
                        [1, 1, 1, 1],
                    ],
                    dtype=np.float64,
                )

                coords = np.matmul(homography_inverse, coords)

                mins = np.min(coords, axis=1)
                maxs = np.max(coords, axis=1)

                xm_bounds = [
                    int(np.floor(np.max([mins[0], 0]))),
                    int(np.ceil(np.min([maxs[0], br_mov.num_x()]))),
                ]
                ym_bounds = [
                    int(np.floor(np.max([mins[1], 0]))),
                    int(np.ceil(np.min([maxs[1], br_mov.num_y()]))),
                ]

                reg_tiles.append((xm_bounds, ym_bounds, xt_bounds, yt_bounds))

                # Get cropping dimensions
                x_crop_bounds = [1024 if xt_bounds[0] > 0 else 0]
                x_crop_bounds.append(
                    2048 + x_crop_bounds[0]
                    if xt_bounds[1] - xt_bounds[0] >= TILE_MERGE_THRESHOLD
                    else xt_bounds[1] - xt_bounds[0] + x_crop_bounds[0],
                )
                y_crop_bounds = [1024 if yt_bounds[0] > 0 else 0]
                y_crop_bounds.append(
                    2048 + y_crop_bounds[0]
                    if yt_bounds[1] - yt_bounds[0] >= TILE_MERGE_THRESHOLD
                    else yt_bounds[1] - yt_bounds[0] + y_crop_bounds[0],
                )
                reg_shape.append((x, y, x_crop_bounds, y_crop_bounds))

                # Start a thread to register the tiles
                threads.append(
                    executor.submit(
                        register_image,
                        br_ref,
                        br_mov,
                        bw,
                        xt_bounds,
                        yt_bounds,
                        xm_bounds,
                        ym_bounds,
                        x,
                        y,
                        x_crop_bounds,
                        y_crop_bounds,
                        max_val,
                        min_val,
                        method,
                        rough_homography_upscaled,
                    ),
                )

                # Bioformats require the first tile be written before any other tile
                if first_tile:
                    logger.info("Waiting for first_tile to finish...")
                    first_tile = False
                    threads[0].result()

        # Wait for threads to finish, track progress
        for thread_num in range(len(threads)):
            if thread_num % 10 == 0:
                pct = 100 * thread_num / len(threads)
                logger.info(f"Registration progress: {pct:6.2f}%")
            reg_homography.append(threads[thread_num].result())

    # Close the image
    bw.close_image()
    logger.info(f"Registration progress: {100.0:6.2f}%")

    # remaining images sharing the moving image's transform
    for moving_image_path in similar_transformation_set:
        # seperate image name from the path to it
        moving_image_name = moving_image_path[-1 * filename_len :]

        logger.info(f"Applying registration to image: {moving_image_name}")

        br_mov = BioReader(moving_image_path, max_workers=read_workers)

        bw = BioWriter(
            str(Path(output_dir).joinpath(moving_image_name)),
            metadata=br_mov.read_metadata(),
            max_workers=write_workers,
        )
        bw.num_x(br_ref.num_x())
        bw.num_y(br_ref.num_y())
        bw.num_z(1)
        bw.num_c(1)
        bw.num_t(1)

        # Apply transformation to remaining images
        logger.info(f"Transformation progress: {0.0:5.2f}%")
        threads = []
        with ThreadPoolExecutor(max_workers=loop_workers) as executor:
            first_tile = True
            for tile, shape, transform in zip(reg_tiles, reg_shape, reg_homography):
                # Start transformation threads
                threads.append(
                    executor.submit(
                        apply_transform,
                        br_mov,
                        bw,
                        tile,
                        shape,
                        transform,
                        method,
                    ),
                )

                # The first tile must be written before all other tiles
                if first_tile:
                    first_tile = False
                    threads[0].result()

            # Wait for threads to finish and track progress
            for thread_num in range(len(threads)):
                if thread_num % 10 == 0:
                    tpct = 100 * thread_num / len(threads)
                    logger.info(f"Transformation progress: {tpct:6.2f}%")
                threads[thread_num].result()
        logger.info(f"Transformation progress: {100.0:6.2f}%")

        bw.close_image()
