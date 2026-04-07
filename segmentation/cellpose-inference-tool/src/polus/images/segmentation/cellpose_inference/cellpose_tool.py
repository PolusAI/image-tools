"""Cellpose Inference Tool - core segmentation logic."""
import logging
import os
import pathlib
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
from typing import Optional

import filepattern as fp
import numpy as np
import torch
from bfio import BioReader
from bfio import BioWriter
from cellpose import models
from cellpose import utils as cp_utils
from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("POLUS_LOG", logging.INFO))

POLUS_IMG_EXT = os.environ.get("POLUS_IMG_EXT", ".ome.tif")

# Number of parallel workers (default 1 — GPU contexts are not fork-safe)
NUM_WORKERS = max(1, int(os.environ.get("NUM_WORKERS", "1")))

# Ruff PLR2004 — named constants for magic literals used in comparisons
_MIN_COMPOUND_EXT_PARTS = 2  # e.g. [".ome", ".tif"]
_MAX_EXT_PART_LEN = 6  # short suffixes like ".ome" vs longer ones
_NORM_PERCENTILE_PARTS = 2  # "low,high" → exactly two values


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _output_path(inp_image: pathlib.Path, out_dir: pathlib.Path) -> pathlib.Path:
    """Derive the output file path for a given input image."""
    if (
        len(inp_image.suffixes) >= _MIN_COMPOUND_EXT_PARTS
        and len(inp_image.suffixes[-2]) < _MAX_EXT_PART_LEN
    ):
        ext = "".join(inp_image.suffixes[-2:])
    else:
        ext = inp_image.suffix
    out_name = inp_image.name.replace(ext, POLUS_IMG_EXT)
    return out_dir / out_name


def _read_2d(
    br: BioReader,
    z: int,
    t: int,
    channel: int,
) -> np.ndarray:
    """Read one Z-plane and return a 2D grayscale array for cellpose v4.

    BioReader stores images as 5D ``(Y, X, Z, C, T)``. A single Z-plane and
    channel are sliced out and squeezed to ``(Y, X)`` — the shape cellpose v4
    expects for a grayscale input. The caller decides whether to pass the
    cytoplasm or nucleus channel index.

    Args:
        br: Open BioReader instance.
        z: Z-index to read.
        t: T-index to read.
        channel: 0-indexed bfio channel to read (cytoplasm or nucleus).

    Returns:
        numpy array of shape ``(Y, X)``.
    """
    return np.squeeze(br[:, :, z : z + 1, channel : channel + 1, t : t + 1])


def _read_3d(
    br: BioReader,
    t: int,
    channel_cyto: int,
    channel_nuc: int,
) -> np.ndarray:
    """Read the full Z-stack for 3-D segmentation.

    In cellpose v4 the ``channels`` parameter is deprecated.  The image is
    returned as ``(Z, Y, X)`` for a single cytoplasm channel, or
    ``(Z, Y, X, 2)`` with channel 0 = cytoplasm and channel 1 = nucleus.

    Returns:
        numpy array shaped (Z, Y, X) or (Z, Y, X, 2).
    """
    cyto_stack = np.squeeze(
        br[:, :, :, channel_cyto : channel_cyto + 1, t : t + 1],
    )  # (Y, X, Z)
    cyto_stack = np.transpose(cyto_stack, (2, 0, 1))  # (Z, Y, X)

    if channel_nuc >= 0 and channel_nuc < br.C and channel_nuc != channel_cyto:
        nuc_stack = np.squeeze(
            br[:, :, :, channel_nuc : channel_nuc + 1, t : t + 1],
        )  # (Y, X, Z)
        nuc_stack = np.transpose(nuc_stack, (2, 0, 1))  # (Z, Y, X)
        return np.stack([cyto_stack, nuc_stack], axis=-1)  # (Z, Y, X, 2)

    return cyto_stack  # (Z, Y, X)


def _parse_norm_percentile(
    value: Optional[str],
) -> Optional[tuple[float, float]]:
    """Parse a 'low,high' string into a (float, float) tuple, or return None."""
    if value is None:
        return None
    parts = value.split(",")
    if len(parts) != _NORM_PERCENTILE_PARTS:
        msg = f"normPercentile must be 'low,high' (e.g. '1.0,99.0'), got: {value!r}"
        raise ValueError(msg)
    return float(parts[0].strip()), float(parts[1].strip())


def _parse_flow3d_smooth(
    value: Optional[str],
) -> float | list[float]:
    """Parse flow3dSmooth: a single float or 'z,y,x' triple."""
    if value is None:
        return 0.0
    parts = [float(v.strip()) for v in value.split(",")]
    return parts[0] if len(parts) == 1 else parts


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def segment_image(  # noqa: PLR0913
    inp_image: pathlib.Path,
    out_dir: pathlib.Path,
    model_type: str = "cpsam",
    diameter: float = 0.0,
    channel_cyto: int = 0,
    channel_nuc: int = -1,
    flow_threshold: float = 0.4,
    cellprob_threshold: float = 0.0,
    use_gpu: bool = False,
    do_3d: bool = False,
    stitch_threshold: float = 0.0,
    min_size: int = 15,
    niter: int = 0,
    anisotropy: float = 1.0,
    exclude_on_edges: bool = False,
    no_norm: bool = False,
    norm_percentile: Optional[str] = None,
    batch_size: int = 8,
    augment: bool = False,
    flow3d_smooth: Optional[str] = None,
    gpu_id: int = 0,
) -> None:
    """Segment a single image file using Cellpose and write a label mask.

    The output is a label image (uint32) where each unique integer value
    corresponds to one detected cell/nucleus. The file is saved in the
    format controlled by the ``POLUS_IMG_EXT`` environment variable
    (default ``.ome.tif``).

    Args:
        inp_image: Path to the input image.
        out_dir: Directory where the output label image is saved.
        model_type: Pretrained model name passed as ``pretrained_model`` to
            ``CellposeModel``. Default ``cpsam`` (SAM-based, v4+).
            Legacy options: ``cyto3``, ``cyto2``, ``cyto``, ``nuclei``,
            ``bact_omni``, ``cyto2_omni``.
        diameter: Expected cell diameter in pixels. ``0.0`` triggers automatic
            estimation by Cellpose.
        channel_cyto: 0-indexed channel in the input image that contains the
            cytoplasm signal (default ``0``).
        channel_nuc: 0-indexed channel in the input image that contains the
            nuclear signal. Use ``-1`` (default) if no dedicated nuclear
            channel is available.
        flow_threshold: Maximum allowed error of the flows for each mask.
            Increase to accept more masks; decrease to be more strict.
            Set to ``0`` to disable this QC step.
        cellprob_threshold: Threshold on the cell-probability output. Lower
            values include more low-confidence detections.
        use_gpu: Whether to use GPU acceleration. Cellpose auto-selects the
            best available backend: CUDA (NVIDIA) → MPS (Apple Silicon) → CPU.
        do_3d: Run full 3-D segmentation across the Z-stack rather than
            processing each plane independently.
        stitch_threshold: When ``do_3d=False``, stitch 2-D masks across
            Z-planes whose IoU exceeds this value (``0.0`` disables stitching).
        min_size: Minimum number of pixels per mask. Masks smaller than this
            are discarded. Set to ``-1`` to disable.
        niter: Number of iterations for the mask-dynamics step. ``0`` sets it
            automatically proportional to the diameter. Increase (e.g. ``2000``)
            for very elongated ROIs.
        anisotropy: Ratio of Z voxel size to XY pixel size. Only used when
            ``do_3d=True``. A value of ``1.0`` assumes isotropic voxels.
        exclude_on_edges: Discard masks that touch the border of the image.
        no_norm: Disable image normalisation. By default Cellpose normalises
            each image to the 1st-99th percentile range.
        norm_percentile: Override the normalisation percentile range as a
            ``"low,high"`` string (e.g. ``"1.0,99.0"``). Ignored when
            ``no_norm=True``.
        batch_size: Number of image tiles to pass through the network in one
            forward pass. Increase on high-VRAM GPUs for speed.
        augment: Tile the image with overlapping tiles and flip overlapping
            regions during inference for slightly improved accuracy.
        flow3d_smooth: Standard deviation of the Gaussian applied to smooth
            the 3-D flow field. Provide a single value (e.g. ``"1.0"``) or a
            comma-separated ZYX triple (e.g. ``"2.0,1.0,1.0"``). ``"0"``
            disables smoothing. Only used when ``do_3d=True``.
        gpu_id: Index of the GPU device to use (e.g. ``0``, ``1``). Only
            relevant when ``use_gpu=True``. Assigned automatically by
            ``batch_segment`` when multiple GPUs are available.
    """
    t0 = time.perf_counter()
    logger.info(f"Processing: {inp_image.name}")

    norm_pct = _parse_norm_percentile(norm_percentile)
    smooth = _parse_flow3d_smooth(flow3d_smooth)

    # Pin this worker to the assigned GPU before creating the model so that
    # each worker in a multi-GPU setup uses a different device.
    if use_gpu and torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        logger.debug(f"Using GPU {gpu_id} for {inp_image.name}")

    # CellposeModel replaces the old Cellpose wrapper in v4.
    # pretrained_model accepts model names ('cpsam', 'cyto3', …) directly.
    model = models.CellposeModel(pretrained_model=model_type, gpu=use_gpu)
    out_path = _output_path(inp_image, out_dir)

    # Build normalize value: False, True, or a dict with percentile overrides.
    # In v4 the normalize param accepts a dict with keys matching normalize_img.
    if no_norm:
        normalize: bool | dict = False
    elif norm_pct is not None:
        normalize = {"normalize": True, "percentile": list(norm_pct)}
    else:
        normalize = True

    # eval() kwargs shared by both 2-D and 3-D paths
    eval_kwargs: dict = {
        "diameter": diameter if diameter > 0 else None,
        "flow_threshold": flow_threshold,
        "cellprob_threshold": cellprob_threshold,
        "min_size": min_size,
        "niter": niter if niter > 0 else None,
        "normalize": normalize,
        "batch_size": batch_size,
        "augment": augment,
    }

    with BioReader(inp_image) as br, BioWriter(out_path, metadata=br.metadata) as bw:
        bw.dtype = np.uint32
        bw.C = 1

        for t in range(br.T):
            if do_3d:
                img = _read_3d(br, t, channel_cyto, channel_nuc)
                # eval returns (masks, flows, styles) in cellpose v4
                masks, _, _ = model.eval(
                    img,
                    do_3D=True,
                    anisotropy=anisotropy,
                    flow3D_smooth=smooth,
                    **eval_kwargs,
                )
                if exclude_on_edges:
                    masks = cp_utils.remove_edge_masks(masks)
                # masks: (Z, Y, X) → bfio expects (Y, X, Z)
                masks_out = np.transpose(masks.astype(np.uint32), (1, 2, 0))
                bw[:, :, :, 0, t] = masks_out
            else:
                for z in range(br.Z):
                    channel = channel_nuc if channel_cyto < 0 else channel_cyto
                    img = _read_2d(br, z, t, channel)
                    masks, _, _ = model.eval(
                        img,
                        stitch_threshold=stitch_threshold,
                        **eval_kwargs,
                    )
                    if exclude_on_edges:
                        masks = cp_utils.remove_edge_masks(masks)
                    bw[:, :, z, 0, t] = masks.astype(np.uint32)

    elapsed = time.perf_counter() - t0
    logger.info(f"Saved: {out_path.name} ({elapsed:.1f}s)")


def batch_segment(  # noqa: PLR0913
    inp_dir: pathlib.Path | list[pathlib.Path],
    out_dir: pathlib.Path,
    file_pattern: Optional[str] = ".+",
    model_type: str = "cpsam",
    diameter: float = 0.0,
    channel_cyto: int = 0,
    channel_nuc: int = -1,
    flow_threshold: float = 0.4,
    cellprob_threshold: float = 0.0,
    use_gpu: bool = False,
    do_3d: bool = False,
    stitch_threshold: float = 0.0,
    min_size: int = 15,
    niter: int = 0,
    anisotropy: float = 1.0,
    exclude_on_edges: bool = False,
    no_norm: bool = False,
    norm_percentile: Optional[str] = None,
    batch_size: int = 8,
    augment: bool = False,
    flow3d_smooth: Optional[str] = None,
    num_workers: int = NUM_WORKERS,
) -> None:
    """Run Cellpose segmentation on a batch of images.

    Discovers input images via ``filepattern`` when *inp_dir* is a directory,
    or processes the provided list of paths directly. Images are dispatched to
    a ``ProcessPoolExecutor`` (size controlled by the ``NUM_WORKERS``
    environment variable; defaults to ``1`` to avoid GPU-context conflicts).

    Args:
        inp_dir: Input directory or explicit list of image paths.
        out_dir: Directory where output label images are written.
        file_pattern: Filepattern expression used to select files inside
            *inp_dir* (ignored when *inp_dir* is a list).
        model_type: Pretrained Cellpose model name.
        diameter: Expected cell diameter in pixels (``0.0`` = auto).
        channel_cyto: 0-indexed cytoplasm channel.
        channel_nuc: 0-indexed nucleus channel (``-1`` = none).
        flow_threshold: Flow error threshold (``0`` = disabled).
        cellprob_threshold: Cell probability threshold.
        use_gpu: Use GPU acceleration (auto-selects CUDA → MPS → CPU).
        do_3d: Enable 3-D segmentation.
        stitch_threshold: IoU threshold for stitching 2-D masks across Z.
        min_size: Minimum pixels per mask (``-1`` = disabled).
        niter: Dynamics iterations (``0`` = auto).
        anisotropy: Z/XY voxel size ratio for 3-D segmentation.
        exclude_on_edges: Discard masks touching image borders.
        no_norm: Disable image normalisation.
        norm_percentile: Normalisation range as ``"low,high"`` string.
        batch_size: Tile batch size for network inference.
        augment: Use tile augmentation during inference.
        flow3d_smooth: 3-D flow smoothing sigma as a single value or
            ``"z,y,x"`` triple.
        num_workers: Number of parallel worker processes. Defaults to the
            ``NUM_WORKERS`` environment variable (default ``1``). Set to the
            number of available GPUs for maximum multi-GPU throughput.
    """
    if isinstance(inp_dir, pathlib.Path):
        pattern = file_pattern or ".+"
        fps = fp.FilePattern(inp_dir, pattern)
        files = [f[1][0] for f in fps()]
    else:
        files = list(inp_dir)

    if not files:
        logger.warning("No files found matching the given pattern.")
        return

    # Detect available GPUs and distribute files across them round-robin.
    # Falls back to gpu_id=0 on CPU-only runs or single-GPU hosts.
    num_gpus = torch.cuda.device_count() if use_gpu else 1
    num_gpus = max(num_gpus, 1)

    logger.info(
        f"Found {len(files)} file(s) to segment | "
        f"workers={num_workers} | "
        f"GPUs={num_gpus if use_gpu else 'CPU'} | "
        f"model={model_type}",
    )
    if use_gpu and num_gpus > 1:
        logger.info(f"Multi-GPU mode: distributing across {num_gpus} GPU(s).")

    failed = 0
    t_start = time.perf_counter()

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(
                segment_image,
                f,
                out_dir,
                model_type,
                diameter,
                channel_cyto,
                channel_nuc,
                flow_threshold,
                cellprob_threshold,
                use_gpu,
                do_3d,
                stitch_threshold,
                min_size,
                niter,
                anisotropy,
                exclude_on_edges,
                no_norm,
                norm_percentile,
                batch_size,
                augment,
                flow3d_smooth,
                idx % num_gpus,
            ): f
            for idx, f in enumerate(files)
        }
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Segmenting images",
            unit="img",
            colour="cyan",
            mininterval=5,
        ):
            try:
                future.result()
            except (OSError, RuntimeError, ValueError) as exc:
                failed += 1
                logger.error(f"Failed to segment {futures[future].name}: {exc}")

    total = time.perf_counter() - t_start
    succeeded = len(files) - failed
    logger.info(
        f"Batch complete: {succeeded}/{len(files)} succeeded, "
        f"{failed} failed | total time: {total:.1f}s "
        f"({total / len(files):.1f}s/image avg)",
    )
