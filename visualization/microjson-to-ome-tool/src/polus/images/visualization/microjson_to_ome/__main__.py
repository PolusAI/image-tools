"""Micojson to Ome."""
import errno
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from pathlib import Path
from typing import Any

import filepattern as fp
import polus.images.visualization.microjson_to_ome.microjson_ome as mo
import typer
from tqdm import tqdm

logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
POLUS_LOG = getattr(logging, os.environ.get("POLUS_LOG", "INFO"))
logger = logging.getLogger("polus.images.visualization.micojson_to_ome")
logger.setLevel(POLUS_LOG)

# Set number of threads for scalability

THREADS = ThreadPoolExecutor()._max_workers

app = typer.Typer(help="Reconstruct binary images from polygon coordinates.")


@app.command()
def main(
    inp_dir: Path = typer.Option(
        ...,
        "--inpDir",
        "-i",
        help="Path to input directory containing binary images.",
    ),
    file_pattern: str = typer.Option(
        ".*",
        "--filePattern",
        "-f",
        help="Filename pattern used to separate data.",
    ),
    out_dir: Path = typer.Option(
        ...,
        "--outDir",
        "-o",
        help="Output collection.",
    ),
    preview: bool = typer.Option(
        False,
        "--preview",
        "-p",
        help="Generate preview of expected outputs.",
    ),
) -> None:
    """Convert binary segmentations to micojson."""
    logger.info(f"inpDir: {inp_dir}")
    logger.info(f"filePattern: {file_pattern}")
    logger.info(f"outDir: {out_dir}")
    starttime = time.time()

    if not inp_dir.exists():
        msg = "inpDir does not exist"
        raise ValueError(msg, inp_dir)

    if not out_dir.exists():
        msg = "outDir does not exist"
        raise ValueError(msg, out_dir)

    fps = fp.FilePattern(inp_dir, file_pattern)

    with ThreadPoolExecutor(max_workers=THREADS) as executor:
        threads = []
        for _, f in enumerate(tqdm(fps())):
            if not f[1][0].name.endswith(".json"):
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), f)
            model = mo.MicrojsonOmeModel(
                out_dir=out_dir,
                file_path=f[1][0],
            )
            future = executor.submit(model.microjson_to_ome)
            threads.append(future)

        for f in tqdm(
            list(as_completed(threads)),
            desc="Reconstruct binary images from polygon coordinates",
            total=len(threads),
        ):
            f.result()

    if preview:
        with Path.open(Path(out_dir, "preview.json"), "w") as jfile:
            out_json: dict[str, Any] = {
                "filepattern": file_pattern,
                "outDir": [],
            }
            for file in fps():
                out_name = f'{file[1][0].name.split("_")[0]!s}.ome.tif'
                out_json["outDir"].append(out_name)
            json.dump(out_json, jfile, indent=2)
        logger.info(f"generating preview data in {out_dir}")

    endtime = (time.time() - starttime) / 60
    logger.info(f"Total time taken for execution: {endtime:.4f} minutes")
    logger.info(f"Total time taken for a single file: {endtime/len(fps):.4f} minutes")


if __name__ == "__main__":
    app()
