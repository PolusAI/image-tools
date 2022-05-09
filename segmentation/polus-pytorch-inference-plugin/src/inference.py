import logging
import pathlib
import typing

import torch

import utils
from data import SCALABLE_BATCH
from data import ScalableDataset

logging.basicConfig(
    format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
    datefmt='%d-%b-%y %H:%M:%S',
)
logger = logging.getLogger("inference")
logger.setLevel(utils.POLUS_LOG)


class ParallelModel:
    def __init__(
            self,
            network: torch.jit.ScriptModule,
            devices: typing.Optional[typing.List[torch.device]],
            output_device: torch.device,
    ):
        self.network = network
        self.output_device = output_device

        if devices is None:
            self.devices = [output_device]
            self.replicas = None
        else:
            self.devices = devices
            self.replicas = torch.nn.parallel.replicate(self.network, devices=devices)

    def infer(self, batch: SCALABLE_BATCH) -> SCALABLE_BATCH:
        if self.replicas is None:
            return self.__infer_cpu(batch)
        else:
            self.__infer_parallel(batch)

    def __infer_parallel(self, batch: SCALABLE_BATCH) -> SCALABLE_BATCH:
        paths, indices, tiles = batch

        inputs = torch.nn.parallel.scatter(tiles, target_gpus=self.devices)
        replicas = self.replicas[:len(inputs)]
        outputs = torch.nn.parallel.parallel_apply(replicas, inputs)
        output = torch.nn.parallel.gather(outputs, target_device=self.output_device)

        return paths, indices, output

    def __infer_cpu(self, batch: SCALABLE_BATCH) -> SCALABLE_BATCH:
        paths, indices, tiles = batch
        outputs = self.network.forward(tiles)
        return paths, indices, outputs


def run_inference(
        model_path: pathlib.Path,
        devices: typing.Optional[typing.List[torch.device]],
        image_paths: typing.List[pathlib.Path],
        output_dir: pathlib.Path,
):
    zarr_paths = {path: utils.get_zarr_path(output_dir, path.name) for path in image_paths}
    [utils.init_zarr_file(in_path, out_path) for (in_path, out_path) in zarr_paths.items()]

    available_memory = utils.get_available_memory(devices)
    tile_memory = 4 * utils.TILE_STRIDE * utils.TILE_STRIDE
    batch_size = max(1, int(available_memory / tile_memory))
    dataset = ScalableDataset(image_paths, utils.TILE_STRIDE, batch_size)

    output_device = devices[0] if devices is not None else torch.device('cpu')
    network = torch.jit.load(model_path, map_location=output_device)
    network.eval()
    model = ParallelModel(network, devices, output_device)

    for i in range(dataset.num_batches()):
        batch = dataset.load_batch(i)
        in_paths, indices, predictions = model.infer(batch)

        out_paths = [zarr_paths[path] for path in in_paths]
        predictions = predictions.cpu().numpy()

        # TODO: Use threading for this
        [utils.write_to_zarr(*args) for args in zip(out_paths, indices, predictions)]

    return
