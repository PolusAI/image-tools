import argparse
import logging
import pathlib

import filepattern
import torch

import inference
import utils

logging.basicConfig(
    format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
    datefmt='%d-%b-%y %H:%M:%S',
)
logger = logging.getLogger('main')
logger.setLevel(utils.POLUS_LOG)


def main(args):
    # TODO: Move into __main__ block after edits

    model_dir = pathlib.Path(args.modelDir).resolve()
    assert model_dir.exists(), f'Path not found: {model_dir}'

    model_path = model_dir.joinpath('model.pth')
    assert model_path.exists(), f'Path not found: {model_path}'

    images_dir = pathlib.Path(args.imagesDir).resolve()
    assert images_dir.exists(), f'Path not found: {images_dir}'
    if images_dir.joinpath('images').is_dir():
        images_dir = images_dir.joinpath('images')

    image_paths = [
        pathlib.Path(file[0]['file']).resolve()
        for file in filepattern.FilePattern(images_dir, str(args.filePattern))()
    ]
    assert len(image_paths) > 0, f'Pattern {args.filePattern} does not match with any images in {images_dir}.'

    device_name = str(args.device)
    if not torch.cuda.is_available():
        devices = None

    elif device_name == 'cpu':
        devices = None

    elif device_name == 'gpu':
        devices = [torch.device('cuda')]

    else:
        if device_name == 'all':
            devices = list(range(torch.cuda.device_count()))
        else:
            devices = list(map(int, device_name.split(',')))
            min_i, max_i = min(devices), max(devices)
            assert min_i >= 0, f'Device indices cannot be negative'
            assert max_i < torch.cuda.device_count(), f'device index {max_i} exceeds the number of GPUs available ({torch.cuda.device_count()})'

        devices = [torch.device(f'cuda:{i}') for i in devices]

    output_dir = pathlib.Path(args.outDir).resolve()
    assert output_dir.exists(), f'Path not found: {output_dir}'

    inference.run_inference(model_path, devices, image_paths, output_dir)

    return


if __name__ == '__main__':
    # multiprocessing.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser(prog='main', description='Segmentation models training plugin')

    parser.add_argument('--modelDir', dest='modelDir', type=str, required=True,
                        help='Directory with a "model.pth" file which can be used to load a model.')
    parser.add_argument('--imagesDir', dest='imagesDir', type=str, required=True,
                        help='Directory with images on which to run inference.')
    parser.add_argument('--filePattern', dest='filePattern', type=str, required=False, default='.+',
                        help='File Pattern for selecting images')
    parser.add_argument('--device', dest='device', type=str, required=False, default='"gpu"',
                        help='Which device(s) to use for running the model.')

    parser.add_argument('--outDir', dest='outDir', type=str, required=True,
                        help='Directory where to store the results of inference.')

    main(args=parser.parse_args())
