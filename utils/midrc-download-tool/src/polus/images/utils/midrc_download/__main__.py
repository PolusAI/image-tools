"""Package entrypoint for the midrc_download package."""

# Base packages
import logging
from os import environ
from pathlib import Path
from typing import Optional

import polus.images.utils.midrc_download.midrc_download as md
import polus.images.utils.midrc_download.utils as ut
import typer

logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
POLUS_LOG = getattr(logging, environ.get("POLUS_LOG", "INFO"))
logger = logging.getLogger("polus.plugins.utils.midrc_download.midrc_download")
logger.setLevel(POLUS_LOG)

POLUS_IMG_EXT = environ.get("POLUS_IMG_EXT", ".ome.tif")

app = typer.Typer(help="Midrc Download.")


@app.command()
def main(  # noqa: PLR0913
    midrc_type: ut.MIDRCTYPES = typer.Option(
        ...,
        "--MidrcType",
        "-mt",
        help="The node_id in the data model utilized in queries and API requests.",
    ),
    project_id: Optional[list[str]] = typer.Option(
        None,
        "--projectId",
        "-p",
        help="The code of the project that this dataset belongs.",
    ),
    sex: Optional[list[str]] = typer.Option(
        None,
        "--sex",
        "-s",
        help="A gender information.",
    ),
    race: Optional[list[str]] = typer.Option(
        None,
        "--race",
        "-r",
        help="Race.",
    ),
    ethnicity: Optional[list[str]] = typer.Option(
        None,
        "--ethnicity",
        "-e",
        help="A racial or cultural background.",
    ),
    age_at_index: Optional[list[str]] = typer.Option(
        None,
        "--ageAtIndex",
        "-a",
        help="The age of the study participant.",
    ),
    study_modality: Optional[list[str]] = typer.Option(
        None,
        "--studyModality",
        "-sm",
        help="The modalities of the imaging study.",
    ),
    body_part_examined: Optional[list[str]] = typer.Option(
        None,
        "--bodyPartExamined",
        "-b",
        help="Body Part Examined.",
    ),
    loinc_contrast: Optional[list[str]] = typer.Option(
        None,
        "--loincContrast",
        "-lc",
        help="The indicator if the image was completed with or without contrast",
    ),
    loinc_method: Optional[list[str]] = typer.Option(
        None,
        "--loincMethod",
        "-lm",
        help="The LOINC method or imaging modality associated with LOINC code.",
    ),
    loinc_system: Optional[list[str]] = typer.Option(
        None,
        "--loincSystem",
        "-ls",
        help="The LOINC system or body part examined associated with LOINC code.",
    ),
    study_year: Optional[list[str]] = typer.Option(
        None,
        "--studyYear",
        "-sy",
        help="The year when imaging study was performed.",
    ),
    covid19_positive: Optional[list[str]] = typer.Option(
        None,
        "--covid19Positive",
        "-c",
        help="An indicator of whether patient has covid infection or not.",
    ),
    source_node: Optional[list[str]] = typer.Option(
        None,
        "--sourceNode",
        "-sn",
        help="A package of image files and metadata related to several imaging series.",
    ),
    data_format: Optional[list[str]] = typer.Option(
        None,
        "--dataFormat",
        "-df",
        help="The file format, physical medium, or dimensions of the resource.",
    ),
    data_category: Optional[list[str]] = typer.Option(
        None,
        "--dataCategory",
        "-dc",
        help="Image files and metadata related to several imaging series.",
    ),
    data_type: Optional[list[str]] = typer.Option(
        None,
        "--dataType",
        "-dt",
        help="The file format, physical medium, or dimensions of the resource.",
    ),
    first: Optional[int] = typer.Option(
        None,
        "--first",
        "-fi",
        help="Number of rows to return.",
    ),
    offset: Optional[int] = typer.Option(
        None,
        "--offset",
        "-of",
        help="Starting position.",
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
) -> None:
    """Midrc Download."""
    logger.info(f"MidrcType: {midrc_type}")
    logger.info(f"projectId: {project_id}")
    logger.info(f"sex: {sex}")
    logger.info(f"race: {race}")
    logger.info(f"ethnicity: {ethnicity}")
    logger.info(f"ageAtIndex: {age_at_index}")
    logger.info(f"studyModality: {study_modality}")
    logger.info(f"bodyPartExamined: {body_part_examined}")
    logger.info(f"loincContrast: {loinc_contrast}")
    logger.info(f"loincMethod: {loinc_method}")
    logger.info(f"loincSystem: {loinc_system}")
    logger.info(f"studyYear: {study_year}")
    logger.info(f"covid19Positive: {covid19_positive}")
    logger.info(f"sourceNode: {source_node}")
    logger.info(f"dataFormat: {data_format}")
    logger.info(f"dataCategory: {data_category}")
    logger.info(f"dataType: {data_type}")
    logger.info(f"first: {first}")
    logger.info(f"offset: {offset}")
    logger.info(f"outDir: {out_dir}")

    option_values = [
        md.cred,
        study_modality,
        loinc_method,
        midrc_type.value,
        loinc_system,
        study_year,
        project_id,
        sex,
        race,
        ethnicity,
        age_at_index,
        loinc_contrast,
        body_part_examined,
        covid19_positive,
        source_node,
        data_format,
        data_category,
        data_type,
        first,
        offset,
        out_dir,
    ]

    params = ut.get_params(option_values)

    if preview:
        ut.generate_preview(out_dir)
        logger.info(f"generating preview data in {out_dir}")
    else:
        model = md.MIDRIC(**params)
        filter_obj = model.get_query(params)

        sort_fields = [{"submitter_id": "asc"}]

        data = model.query_data(
            midrc_type=midrc_type.value,
            fields=None,
            filter_object=filter_obj,
            sort_fields=sort_fields,
            first=first,
            offset=offset,
        )
        model.download_data(data)


if __name__ == "__main__":
    app()
