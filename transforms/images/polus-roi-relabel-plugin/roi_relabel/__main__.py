import argparse
import concurrent.futures
import pathlib
import time

from roi_relabel import methods
from roi_relabel.utils import constants
from roi_relabel.utils import helpers

helpers.configure_logging()
logger = helpers.make_logger('main')

""" Define the arguments.
"""
logger.info("Parsing arguments...")
parser = argparse.ArgumentParser(
    prog='main',
    description='Relabel RoIs in an image collection.',
)

parser.add_argument(
    '--inpDir', dest='inpDir', type=str, required=True,
    help='Path to input collection.',
)

parser.add_argument(
    '--method', dest='method', type=str, required=True, choices=methods.METHODS,
    help='What operation to perform on the images.',
)

parser.add_argument(
    '--outDir', dest='outDir', type=str, required=True,
    help='Path to output collection.',
)

""" Parse the arguments.
"""
args = parser.parse_args()

inp_dir = pathlib.Path(args.inpDir).resolve()
assert inp_dir.exists(), f'Path not found {inp_dir}'
if inp_dir.joinpath('images').is_dir():
    inp_dir = inp_dir.joinpath('images')

method_name = args.method

out_dir = pathlib.Path(args.outDir).resolve()
assert out_dir.exists(), f'Path not found {out_dir}'

logger.info(f'inpDir = {inp_dir}')
logger.info(f'method = {method_name}')
logger.info(f'outDir = {out_dir}')

""" Relabel the images.
"""
helpers.seed_everything(42)
image_paths = [path for path in inp_dir.iterdir() if '.ome.' in path.name]

start = time.perf_counter()
logger.info(f'Getting started ...')

with concurrent.futures.ProcessPoolExecutor(max_workers=constants.NUM_THREADS) as executor:
    futures: list[concurrent.futures.Future[bool]] = list()

    for inp_path in image_paths:
        out_path = out_dir.joinpath(helpers.replace_extension(inp_path.name))
        futures.append(executor.submit(
            methods.relabel,
            inp_path,
            out_path,
            method_name,
        ))

    done, not_done = concurrent.futures.wait(futures, 10)
    while len(not_done) > 0:
        logger.info(f'Progress {100 * len(done) / len(futures):6.2f}% ...')
        done, not_done = concurrent.futures.wait(futures, 10)

end = time.perf_counter()
logger.info(f'Finished relabeling {len(image_paths)} images in {end - start:.2e} seconds!')
