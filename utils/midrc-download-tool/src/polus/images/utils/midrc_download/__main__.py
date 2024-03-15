"""Package entrypoint for the midrc_download package."""

# Base packages
import json
import logging
from os import environ
from pathlib import Path
from utils import *
from typing import Union
from typing import List, Optional
from typing import Any

import typer
import polus.images.utils.midrc_download.midrc_download as md
import polus.images.utils.midrc_download.utils as ut


logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
POLUS_LOG = getattr(logging, environ.get("POLUS_LOG", "INFO"))
logger = logging.getLogger("polus.plugins.utils.midrc_download.midrc_download")
logger.setLevel(POLUS_LOG)

POLUS_IMG_EXT = environ.get("POLUS_IMG_EXT", ".ome.tif")

app = typer.Typer(help="Midrc Download.")

# def generate_preview(
#     img_path: Path,
#     out_dir: Path,
# ) -> None:
#     """Generate preview of the plugin outputs."""

#     preview = {}

#     with Path.open(out_dir / "preview.json", "w") as fw:
#         json.dump(preview, fw, indent=2)


@app.command()
def main(
    data_type: dataType = typer.Option(
        ...,
        "--dataType",
        "-d",
        help="projectId.",
    ),
    project_id: Optional[List[str]]= typer.Option(
        None,
        "--projectId",
        "-p",
        help="projectId.",
    ),
    sex: Optional[List[str]]= typer.Option(
        None,
        "--sex",
        "-s",
        help="Sex.",
    ),
    race: Optional[List[str]]= typer.Option(
        None,
        "--race",
        "-r",
        help="Race.",
    ),
    ethnicity: Optional[List[str]]= typer.Option(
        None,
        "--ethnicity",
        "-e",
        help="Ethnicity.",
    ),
    min_age: Optional[int]= typer.Option(
        0,
        "--minAge",
        "-miA",
        help="minAge.",
    ),
    max_age: Optional[int]= typer.Option(
        89,
        "--maxAge",
        "-mxA",
        help="minAge.",
    ),
    study_modality: Optional[List[str]]= typer.Option(
       None,
        "--studyModality",
        "-sm",
        help="studyModality.",
    ),
    body_part_examined: Optional[List[str]]= typer.Option(
        None,
        "--bodyPartExamined",
        "-b",
        help="bodyPartExamined.",
    ),
    loinc_contrast: Optional[List[str]]= typer.Option(
        None,
        "--loincContrast",
        "-lc",
        help="loincContrast.",
    ),
    loinc_method: Optional[List[str]]= typer.Option(
        None,
        "--loincMethod",
        "-lm",
        help="loincMethod.",
    ),
    loinc_system: Optional[List[str]]= typer.Option(
       None,
        "--loincSystem",
        "-ls",
        help="loincSystem.",
    ),
    study_year: Optional[int]= typer.Option(
       None,
        "--studyYear",
        "-sy",
        help="studyYear.",
    ),
    covid19_positive: Optional[List[str]]= typer.Option(
        None,
        "--covid19Positive",
        "-c",
        help="covid19Positive.",
    ),
    first: Optional[int]= typer.Option(
        None,
        "--first",
        "-f",
        help="first.",
    ),
    offset: Optional[int]= typer.Option(
        None,
        "--offset",
        "-o",
        help="offset.",
    ),
    out_dir: Path = typer.Option(
        ...,
        "--outDir",
        "-o",
        help="Output directory.",
        exists=True,
        writable=True,
        file_okay=False,
        resolve_path=True,
    ),
    preview: bool = typer.Option(
        False,
        "--preview",
        "-v",
        help="Preview of expected outputs (dry-run)",
        show_default=False,
    ),
):
    """Midrc Download."""
    logger.info(f"dataType: {data_type}")
    logger.info(f"projectId: {project_id}")
    logger.info(f"sex: {sex}")
    logger.info(f"race: {race}")
    logger.info(f"ethnicity: {ethnicity}")
    logger.info(f"minAge: {min_age}")
    logger.info(f"maxAge: {max_age}")
    logger.info(f"studyModality: {study_modality}")
    logger.info(f"bodyPartExamined: {body_part_examined}")
    logger.info(f"loincContrast: {loinc_contrast}")
    logger.info(f"loincMethod: {loinc_method}")
    logger.info(f"loincSystem: {loinc_system}")
    logger.info(f"studyYear: {study_year}")
    logger.info(f"covid19Positive: {covid19_positive}")
    logger.info(f"first: {first}")
    logger.info(f"offset: {offset}")
    logger.info(f"outDir: {out_dir}")



   
    option_values = [md.cred, study_modality, loinc_method, data_type.value,
                   loinc_system, study_year,project_id, sex, race, ethnicity, min_age, 
                   max_age, loinc_contrast, body_part_examined, covid19_positive, 
                   first, offset, out_dir
                   ]
    params = get_params(option_values)

    model = md.MIDRIC_download(**params)
    filter_obj = model.get_query(params)

    sort_fields=[{"submitter_id": "asc"}]
    data = model.download_request( data_type=data_type.value,
                        fields=None,
                        filter_object=filter_obj,
                        sort_fields=sort_fields,
                        first=first,
                        offset=offset
                        )
    

    study_uids = [i['study_uid'] for i in data]

    filter_object={
                        "AND": [
                            {"IN": {"study_uid": study_uids}},
                            {"IN": {"source_node": ut.SOURCE_NODE}},
                        ]
                    }
    
    data_file = model.download_request(data_type="data_file",
                        fields=None,
                        filter_object=filter_obj,
                        sort_fields=sort_fields,
                        first=first,
                        offset=offset
                   )
    
    model.download_data(data)
    

    
    # if len(data ) > 0:
    #     object_ids = [i['object_id'] for i in data  if 'object_id' in i] ## make a list of the file object_ids returned by our query
    #     print("Query returned {} data files with {} object_ids.".format(len(data),len(object_ids)))
    #     print("Data is a list with rows like this:\n\t {}".format(data))
    # else:
    #     print("Your query returned no data! Please, check that query parameters are valid.")

    # data = model.raw_data_download(
    #                     data_type=data_type.value,
    #                     fields=None,
    #                     filter_object=filter_obj)

    

    
   
    

    # if preview:
    #     generate_preview(inp_dir, out_dir)
    #     logger.info(f"generating preview data in : {out_dir}.")
    #     return

    # midrc_download(inp_dir, filepattern, out_dir)


if __name__ == "__main__":
    app()
