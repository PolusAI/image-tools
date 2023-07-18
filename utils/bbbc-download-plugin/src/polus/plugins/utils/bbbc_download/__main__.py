import json
import os
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Optional

import typer
from tqdm import tqdm
from polus.plugins.utils.bbbc_download.BBBC_model import BBBC, BBBCDataset, IDAndSegmentation, PhenotypeClassification, ImageBasedProfiling
from sys import platform
from multiprocessing import cpu_count



if platform == "linux" or platform == "linux2":
    NUM_THREADS = len(os.sched_getaffinity(0))  # type: ignore
else:
    NUM_THREADS = max(cpu_count() // 2, 2)

app = typer.Typer()

# Initialize the logger
logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger("polus.plugins.utils.bbbc_download")
logger.setLevel(os.environ.get("POLUS_LOG", logging.INFO))

@app.command()
def main(
    name: str= typer.Option(
    ..., "--name", help="The name of the dataset that is to be downloaded"
    ),
    out_dir: Path= typer.Option(
    ...,"--outDir", help="The path for downloading the dataset"
    )

)-> None:
    """Download the required dataset from the BBBC dataaset."""
    logger.info(f"name = {name}")
    logger.info(f"outDir = {out_dir}")
    """Checking if output directory exists. If it does not exist then a designated path is created."""
    if not out_dir.exists():
        out_dir.mkdir()
        logger.info(f"{out_dir} did not exists. Creating new path.")

    with ProcessPoolExecutor(max_workers=NUM_THREADS) as executor:
        threads=[]
        names=name.split(",")
        for n in names:
            if(n=='IDAndSegmentation'):
                threads.append(
                    executor.submit(IDAndSegmentation.raw,out_dir)
                    )
            
            elif(n=='PhenotypeClassification'):
                threads.append(
                    executor.submit(PhenotypeClassification.raw,out_dir)
                    )



            elif(n=='ImageBasedProfiling'):
                threads.append(
                    executor.submit(ImageBasedProfiling.raw,out_dir)
                    )
            
            elif(n=='BBBC'):
                threads.append(
                    executor.submit(BBBC.raw,out_dir)
                    )
                

            else:
                d=executor.submit(BBBCDataset.create_dataset, n)
                d_name=d.result()
                threads.append(
                     executor.submit(d_name.raw,out_dir)
                )

            
        for f in tqdm(
            as_completed(threads),
            total=len(threads),
            mininterval=5,
            desc=f"donwloading the dataset",
            initial=0,
            unit_scale=True,
            colour="cyan",
        ):
            f.result()
        
                
        


if __name__ == "__main__":
    app()