"""Pytest configuration and shared fixtures for Cellpose Inference Tool tests."""

import pathlib
import shutil
import tempfile
import typing

import numpy as np
import pytest
import skimage.draw
import skimage.io

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DEFAULT_DTYPE: np.dtype = np.dtype("uint16")

# Cell geometry used in both fixtures and tests so diameter always matches.
CELL_RADIUS_PX: int = 15  # radius of each synthetic cell in pixels
CELL_DIAMETER_PX: float = float(CELL_RADIUS_PX * 2)


def _make_cell_image(
    size: int = 128,
    n_cells: int = 3,
    radius: int = CELL_RADIUS_PX,
    dtype: np.dtype = _DEFAULT_DTYPE,
) -> np.ndarray:
    """Create a synthetic fluorescence image with bright circular cells.

    Cells are placed on a near-zero background with a small amount of
    Gaussian noise so that the image looks plausible after normalisation.
    Each cell is a filled disk at full-scale intensity, giving cellpose
    strong, unambiguous signal.
    """
    rng = np.random.default_rng(seed=42)
    max_val = int(np.iinfo(dtype).max if np.issubdtype(dtype, np.integer) else 1.0)

    # Low-level background noise
    img = (rng.normal(0, max_val * 0.01, (size, size))).clip(0).astype(dtype)

    # Place non-overlapping cells well inside the image border
    margin = radius + 2
    placed: list[tuple[int, int]] = []
    attempts = 0
    while len(placed) < n_cells and attempts < 1000:
        cy = int(rng.integers(margin, size - margin))
        cx = int(rng.integers(margin, size - margin))
        # Reject if too close to an already-placed cell
        if all(
            ((cy - py) ** 2 + (cx - px) ** 2) > (2 * radius + 2) ** 2
            for py, px in placed
        ):
            rr, cc = skimage.draw.disk((cy, cx), radius, shape=img.shape)
            img[rr, cc] = int(max_val * 0.9)
            placed.append((cy, cx))
        attempts += 1

    return img


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(params=[(128, ".tif")])
def get_params(request: pytest.FixtureRequest) -> tuple[int, str]:
    """Parametrize over image sizes and extensions."""
    return request.param  # type: ignore[return-value]


@pytest.fixture()
def synthetic_images(
    get_params: tuple[int, str],
) -> typing.Generator[tuple[list[np.ndarray], pathlib.Path], None, None]:
    """Generate a small directory of synthetic cell images."""
    size, ext = get_params
    inp_dir = pathlib.Path(tempfile.mkdtemp(suffix="_syn_inp"))
    images: list[np.ndarray] = []

    for i in range(2):
        arr = _make_cell_image(size=size)
        out_path = inp_dir / f"syn_image_{i}{ext}"
        skimage.io.imsave(str(out_path), arr)
        images.append(arr)

    yield images, inp_dir
    shutil.rmtree(inp_dir)


@pytest.fixture()
def output_directory() -> typing.Generator[pathlib.Path, None, None]:
    """Provide a temporary output directory, cleaned up after the test."""
    out_dir = pathlib.Path(tempfile.mkdtemp(suffix="_out"))
    yield out_dir
    shutil.rmtree(out_dir)
