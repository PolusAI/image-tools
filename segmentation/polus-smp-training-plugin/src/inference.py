import pathlib
import typing

import bfio
import numpy
import torch
from segmentation_models_pytorch.base import SegmentationModel
from tqdm import tqdm

from utils import Dataset

TILE_SIZE = 512


def run_inference(
        *,
        model: SegmentationModel,
        device: torch.device,
        image_paths: typing.List[pathlib.Path],
        output_dir: pathlib.Path,
):
    model.eval()

    # TODO(Najib): Fixed batch size to 1 for now. Need to fix this later
    for image_path in tqdm(image_paths, desc=f'running inference on {len(image_paths)} images'):
        with bfio.BioReader(image_path) as reader:
            metadata = reader.metadata
            full_prediction = numpy.zeros(shape=reader.shape, dtype=numpy.float32)

            for z in range(reader.Z):

                for y_min in range(0, reader.Y, TILE_SIZE):
                    y_max = min(reader.Y, y_min + TILE_SIZE)

                    for x_min in range(0, reader.X, TILE_SIZE):
                        x_max = min(reader.X, x_min + TILE_SIZE)

                        tile = numpy.squeeze(reader[y_min:y_max, x_min:x_max, z, 0, 0])
                        if tile.shape != (TILE_SIZE, TILE_SIZE):
                            y, x = tile.shape
                            padded_tile = numpy.zeros(shape=(TILE_SIZE, TILE_SIZE), dtype=tile.dtype)
                            padded_tile[:y, :x] = tile[:]
                            tile = padded_tile
                        else:
                            y = x = TILE_SIZE

                        tile = Dataset.preprocessing(tile.astype(numpy.float32))
                        tile = tile[None, ...]

                        with torch.no_grad():
                            prediction = model.forward(tile.to(device)).cpu()
                        prediction = numpy.squeeze(numpy.asarray(prediction, dtype=full_prediction.dtype))

                        full_prediction[y_min:y_max, x_min:x_max, z, 0, 0] = prediction[:y, :x]

        with bfio.BioWriter(output_dir.joinpath(image_path.name), metadata=metadata) as writer:
            # TODO(Najib): Discuss what the best output dtype would be
            writer.dtype = full_prediction.dtype
            writer[:] = full_prediction[:]

    return
