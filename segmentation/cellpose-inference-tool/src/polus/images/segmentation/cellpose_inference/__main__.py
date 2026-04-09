"""Cellpose Inference Tool - CLI entry point."""
import json
import logging
import os
import pathlib
from typing import Optional

import typer
from polus.images.segmentation.cellpose_inference.cellpose_tool import POLUS_IMG_EXT
from polus.images.segmentation.cellpose_inference.cellpose_tool import batch_segment

app = typer.Typer()

logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger("polus.images.segmentation.cellpose_inference")
logger.setLevel(os.environ.get("POLUS_LOG", logging.INFO))


def _parse_inp_dir(value: str) -> pathlib.Path:
    path = pathlib.Path(value).expanduser().resolve()
    if not path.is_dir():
        msg = f"Directory does not exist: {path}"
        raise typer.BadParameter(msg)
    return path


def _write_preview(
    out_dir: pathlib.Path,
    file_pattern: str,
    files: list[pathlib.Path],
) -> None:
    preview = {
        "filePattern": file_pattern,
        "outDir": [f.stem + POLUS_IMG_EXT for f in files],
    }
    preview_path = out_dir / "preview.json"
    with preview_path.open("w") as fh:
        json.dump(preview, fh, indent=2)
    logger.info(f"Preview written to {preview_path}")


@app.command()
def main(  # noqa: PLR0913
    inp_dir: str = typer.Option(
        ...,
        "--inpDir",
        help="Input image collection to be segmented.",
    ),
    file_pattern: str = typer.Option(
        ".+",
        "--filePattern",
        help="Filepattern to select images inside inpDir.",
    ),
    out_dir: pathlib.Path = typer.Option(
        ...,
        "--outDir",
        help="Output directory for label masks.",
        exists=True,
        resolve_path=True,
        writable=True,
        file_okay=False,
        dir_okay=True,
    ),
    model_type: str = typer.Option(
        "cpsam",
        "--modelType",
        help=(
            "Pretrained Cellpose model. Default: cpsam (SAM-based, v4+). "
            "Legacy options: cyto3, cyto2, cyto, nuclei, bact_omni, cyto2_omni."
        ),
    ),
    channel_cyto: int = typer.Option(
        0,
        "--channelCyto",
        help="0-indexed channel for cytoplasm signal (default 0).",
        min=0,
    ),
    channel_nuc: int = typer.Option(
        -1,
        "--channelNuc",
        help="0-indexed channel for nucleus signal. -1 = no nucleus channel.",
        min=-1,
    ),
    diameter: float = typer.Option(
        0.0,
        "--diameter",
        help="Expected cell diameter in pixels. 0 = automatic estimation.",
        min=0.0,
    ),
    min_size: int = typer.Option(
        15,
        "--minSize",
        help="Minimum number of pixels per mask. -1 disables the filter.",
        min=-1,
    ),
    flow_threshold: float = typer.Option(
        0.4,
        "--flowThreshold",
        help="Maximum allowed flow error for accepted masks. 0 disables this QC step.",
    ),
    cellprob_threshold: float = typer.Option(
        0.0,
        "--cellprobThreshold",
        help="Cell-probability threshold. Lower values keep more detections.",
    ),
    niter: int = typer.Option(
        0,
        "--niter",
        help=(
            "Number of iterations for the mask-dynamics step. "
            "0 = auto (proportional to diameter). "
            "Increase (e.g. 2000) for very elongated ROIs."
        ),
        min=0,
    ),
    do_3d: bool = typer.Option(
        False,
        "--do3D",
        help="Run full 3-D segmentation across the Z-stack.",
    ),
    stitch_threshold: float = typer.Option(
        0.0,
        "--stitchThreshold",
        help=(
            "IoU threshold for stitching 2-D masks across Z-planes. "
            "0 = disabled (only used when --do3D is False)."
        ),
        min=0.0,
        max=1.0,
    ),
    anisotropy: float = typer.Option(
        1.0,
        "--anisotropy",
        help=(
            "Ratio of Z voxel size to XY pixel size. "
            "Only used with --do3D. 1.0 = isotropic voxels."
        ),
        min=0.0,
    ),
    flow3d_smooth: Optional[str] = typer.Option(
        None,
        "--flow3dSmooth",
        help=(
            "Gaussian sigma for smoothing 3-D flow fields. "
            "Single value (e.g. '1.0') or ZYX triple (e.g. '2.0,1.0,1.0'). "
            "0 = disabled. Only used with --do3D."
        ),
    ),
    no_norm: bool = typer.Option(
        False,
        "--noNorm",
        help="Disable image normalisation.",
    ),
    norm_percentile: Optional[str] = typer.Option(
        None,
        "--normPercentile",
        help=(
            "Normalisation percentile range as 'low,high' (e.g. '1.0,99.0'). "
            "Ignored when --noNorm is set."
        ),
    ),
    batch_size: int = typer.Option(
        8,
        "--batchSize",
        help=(
            "Number of image tiles per network forward pass. "
            "Increase on high-VRAM GPUs."
        ),
        min=1,
    ),
    augment: bool = typer.Option(
        False,
        "--augment",
        help="Tile image with overlapping tiles and flip for augmented inference.",
    ),
    use_gpu: bool = typer.Option(
        False,
        "--useGpu",
        help=(
            "Use GPU acceleration. Cellpose auto-selects the best available "
            "backend: CUDA (NVIDIA) → MPS (Apple Silicon) → CPU."
        ),
    ),
    exclude_on_edges: bool = typer.Option(
        False,
        "--excludeOnEdges",
        help="Discard masks that touch the border of the image.",
    ),
    preview: Optional[bool] = typer.Option(
        False,
        "--preview",
        help="Write a preview.json of expected outputs without running segmentation.",
    ),
) -> None:
    """Run Cellpose cell-segmentation on a collection of images.

    Reads images from *inpDir*, segments each one with the chosen Cellpose
    model, and writes uint32 label masks to *outDir*.  The output file format
    is controlled by the ``POLUS_IMG_EXT`` environment variable
    (default ``.ome.tif``).
    """
    inp_path = _parse_inp_dir(inp_dir)

    logger.info(f"inpDir            = {inp_path}")
    logger.info(f"filePattern       = {file_pattern}")
    logger.info(f"outDir            = {out_dir}")
    logger.info(f"modelType         = {model_type}")
    logger.info(f"diameter          = {diameter}")
    logger.info(f"channelCyto       = {channel_cyto}")
    logger.info(f"channelNuc        = {channel_nuc}")
    logger.info(f"flowThreshold     = {flow_threshold}")
    logger.info(f"cellprobThreshold = {cellprob_threshold}")
    logger.info(f"minSize           = {min_size}")
    logger.info(f"niter             = {niter}")
    logger.info(f"do3D              = {do_3d}")
    logger.info(f"stitchThreshold   = {stitch_threshold}")
    logger.info(f"anisotropy        = {anisotropy}")
    logger.info(f"flow3dSmooth      = {flow3d_smooth}")
    logger.info(f"noNorm            = {no_norm}")
    logger.info(f"normPercentile    = {norm_percentile}")
    logger.info(f"batchSize         = {batch_size}")
    logger.info(f"augment           = {augment}")
    logger.info(f"useGpu            = {use_gpu}")
    logger.info(f"excludeOnEdges    = {exclude_on_edges}")
    logger.info(f"preview           = {preview}")

    if preview:
        import filepattern as fp_lib

        fps = fp_lib.FilePattern(inp_path, file_pattern)
        files = [f[1][0] for f in fps()]
        _write_preview(out_dir, file_pattern, files)
        return

    batch_segment(
        inp_dir=inp_path,
        out_dir=out_dir,
        file_pattern=file_pattern,
        model_type=model_type,
        diameter=diameter,
        channel_cyto=channel_cyto,
        channel_nuc=channel_nuc,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        use_gpu=use_gpu,
        do_3d=do_3d,
        stitch_threshold=stitch_threshold,
        min_size=min_size,
        niter=niter,
        anisotropy=anisotropy,
        exclude_on_edges=exclude_on_edges,
        no_norm=no_norm,
        norm_percentile=norm_percentile,
        batch_size=batch_size,
        augment=augment,
        flow3d_smooth=flow3d_smooth,
    )


if __name__ == "__main__":
    app()
