import pathlib
import typing

import bfio
import numpy
import torch
from segmentation_models_pytorch.base import SegmentationModel
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from utils import Dataset, Tile, UnTile

# TILE_SIZE must be a multiple of 1024
TILE_SIZE = 2048
MODEL_TILE_SIZE = 512
BATCH_SIZE = 5


def thread_loader(image_path, device):

    with bfio.BioReader(image_path) as reader:

        image = reader[:]

    image = Dataset.preprocessing(image.astype(numpy.float32)).to(device)
    return image


def thread_save(image_path, output_dir: pathlib.Path, prediction, i):

    with bfio.BioReader(image_path) as reader:

        with bfio.BioWriter(
            output_dir.joinpath(image_path.name), metadata=reader.metadata
        ) as writer:

            writer.dtype = numpy.float32
            writer[:] = prediction[i, 0, :-1, :-1].cpu().numpy()


def run_inference(
    *,
    model: SegmentationModel,
    device: torch.device,
    image_paths: typing.List[pathlib.Path],
    output_dir: pathlib.Path,
):

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    tile = Tile(tile_size=(MODEL_TILE_SIZE, MODEL_TILE_SIZE))
    untile = UnTile(tile_size=(MODEL_TILE_SIZE, MODEL_TILE_SIZE))
    batch_size = torch.cuda.device_count() * BATCH_SIZE

    model.eval()

    # TODO(Najib): Fixed batch size to 1 for now. Need to fix this later
    load_threads = []
    save_threads = []
    batches = []

    for i, image_path in enumerate(image_paths):
        if not i % batch_size:
            batches.append([])
        batches[-1].append(image_path)

    with ThreadPoolExecutor() as executor:
        for image_path in batches[0]:
            load_threads.append(executor.submit(thread_loader, image_path, device))
        paths = batches[0]

        for batch in tqdm(
            batches[1:], desc=f"running inference on {len(image_paths)} images"
        ):
            # Load the data
            patch = torch.stack([t.result() for t in load_threads], axis=0)

            # Start loading the next data
            load_threads = []
            for image_path in batches[0]:
                load_threads.append(executor.submit(thread_loader, image_path, device))

            patch, shape = tile(patch)

            with torch.no_grad():
                prediction = model.forward(patch.to(device))
                prediction = untile(prediction, shape)

            for t in save_threads:
                t.result()
            save_threads = []

            for i in range(len(paths)):
                save_threads.append(
                    executor.submit(thread_save, paths[i], output_dir, prediction, i)
                )

            paths = batch

        # Process the last batch
        patch = torch.stack([t.result() for t in load_threads], axis=0)

        patch, shape = tile(patch)

        with torch.no_grad():
            prediction = model.forward(patch.to(device))
            prediction = untile(prediction, shape).cpu().numpy()[:, 0, :-1, :-1]

        for t in save_threads:
            t.result()
        save_threads = []

        for i in range(len(paths)):
            save_threads.append(
                executor.submit(thread_save, paths[i], output_dir, prediction, i)
            )

        for t in save_threads:
            t.result()

    return
