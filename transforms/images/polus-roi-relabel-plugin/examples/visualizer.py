import os
import pathlib
import tempfile

import bfio
import numpy
import streamlit as st
from matplotlib import pyplot

import roi_relabel.methods

DATA_ROOT = os.environ.get("DATA_ROOT", pathlib.Path(__file__).resolve().parent.parent.joinpath('data', 'tissuenet', 'standard', 'val'))


def read_image(path: pathlib.Path, *, num_channels: int) -> numpy.ndarray:
    with bfio.BioReader(path) as reader:
        image = numpy.squeeze(reader[:, :, 0, 0:num_channels, 0])
    return numpy.squeeze(image)


def draw_plots(
        image_path: pathlib.Path,
        temp_dir: pathlib.Path,
):
    images: dict[str, numpy.ndarray] = {
        'original': read_image(image_path, num_channels=1),
    }
    mask = numpy.asarray((images['original'] > 0), dtype=bool)

    for method in roi_relabel.methods.METHODS:
        if method == 'optimizedGraphColoring':
            continue

        out_path = pathlib.Path(temp_dir).joinpath(f'{method}_{image_path.name}')
        roi_relabel.methods.relabel(image_path, out_path, method, range_multiplier=2.0)
        images[method] = read_image(out_path, num_channels=1)

    pyplot.clf()
    ax_00: pyplot.Axes
    ax_01: pyplot.Axes
    ax_10: pyplot.Axes
    ax_11: pyplot.Axes
    ax_20: pyplot.Axes
    ax_21: pyplot.Axes

    fig, ((ax_00, ax_01), (ax_10, ax_11), (ax_20, ax_21)) = pyplot.subplots(3, 2, figsize=(10, 15), dpi=300)

    axs = [ax_00, ax_01, ax_10, ax_11, ax_20, ax_21]
    for ax, (name, img) in zip(axs, images.items()):
        img = img.astype(numpy.float32)
        img[~mask] = numpy.nan
        ax.imshow(img, origin='upper', cmap='tab20b')
        ax.set_title(name)

    _ = [ax.axis('off') for ax in axs]

    st.pyplot(fig)

    return


def main():

    st.title('RoI Relabeling')

    input_dir = DATA_ROOT.joinpath('labels').resolve()
    assert input_dir.exists(), f'Path not found: {input_dir}'

    names = list(sorted(map(
        lambda path: path.name,
        filter(
            lambda path: '.ome.' in path.name,
            input_dir.iterdir(),
        ),
    )))

    image_name: str = st.selectbox('Image Name:', options=names, index=0)
    with tempfile.TemporaryDirectory() as temp_dir:
        draw_plots(input_dir.joinpath(image_name), temp_dir)

    return


if __name__ == '__main__':
    assert DATA_ROOT.exists(), f'Path not found: {DATA_ROOT}.'
    main()
