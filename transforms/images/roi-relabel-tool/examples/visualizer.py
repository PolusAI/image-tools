"""Visualize RoIs with streamlit."""

import os
import pathlib

import bfio
import helpers
import numpy
import streamlit as st
from matplotlib import pyplot
from matplotlib.figure import Figure
from polus.plugins.transforms.images.roi_relabel import methods
from skimage.measure import label

DATA_ROOT = pathlib.Path(
    os.environ.get(
        "DATA_ROOT",
        pathlib.Path(__file__).resolve().parent.parent.joinpath("data"),
    ),
).resolve()


def gen_image() -> tuple[pathlib.Path, pathlib.Path]:
    """Generate an image for visualizations."""
    DATA_ROOT.mkdir(exist_ok=True)
    return helpers.gen_image(
        length=2048,
        n=4,
        inp_dir=DATA_ROOT.joinpath("inp_dir"),
        out_dir=DATA_ROOT.joinpath("out_dir"),
    )


def read_image(path: pathlib.Path, *, num_channels: int) -> numpy.ndarray:
    """Read the image with bfio."""
    with bfio.BioReader(path) as reader:
        image = numpy.squeeze(reader[:, :, 0, 0:num_channels, 0])
    return numpy.squeeze(image)


def plot_image(
    image: numpy.ndarray,
    mask: numpy.ndarray,
    *,
    name: str = "",
) -> Figure:
    """Plot an image in the given axis."""
    image_ = image.astype(numpy.float32)
    image_[~mask] = numpy.nan
    fig = pyplot.figure(figsize=(6, 6), dpi=300)

    # get the axes out of the figure
    ax = fig.add_subplot(111)
    ax.imshow(image_, origin="upper", cmap="turbo")
    ax.set_title(name)
    ax.axis("off")
    return ax


def draw_plots(
    image_path: pathlib.Path,
    out_dir: pathlib.Path,
) -> None:
    """Draw plots in streamlit."""
    original = read_image(image_path, num_channels=1)
    mask: numpy.ndarray = (original > 0).astype(bool)

    expected: numpy.ndarray = label(original)

    names = [
        ("original", None),
        ("expectedContiguous", None),
        ("contiguous", methods.Methods.contiguous),
        ("randomize", methods.Methods.randomize),
        ("randomByte", methods.Methods.random_byte),
        ("graphColoring", methods.Methods.graph_coloring),
    ]

    for name, method in names:
        if name == "original":
            ax = plot_image(original, mask, name=name)
        elif name == "expectedContiguous":
            ax = plot_image(expected, mask, name=name)
        else:
            out_path = out_dir.joinpath(f"{name}_{image_path.name}")
            methods.relabel(
                image_path,
                out_path,
                method,
                range_multiplier=5.0,
            )
            out_image = read_image(out_path, num_channels=1)
            ax = plot_image(out_image, mask, name=name)
        st.pyplot(ax.figure)


def main() -> None:
    """Run in streamlit."""
    st.title("RoI Relabeling")

    inp_dir, out_dir = gen_image()

    names = sorted(
        path.name
        for path in filter(
            lambda p: p.name.endswith(".ome.tif"),
            inp_dir.iterdir(),
        )
    )

    image_name: str = st.selectbox("Image Name:", options=names, index=0)
    draw_plots(inp_dir.joinpath(image_name), out_dir)


if __name__ == "__main__":
    assert DATA_ROOT.exists(), f"Path not found: {DATA_ROOT}."  # noqa: S101
    main()
