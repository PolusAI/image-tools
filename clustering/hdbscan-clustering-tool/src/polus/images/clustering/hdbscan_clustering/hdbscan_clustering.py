"""Hdbscan Clustering Plugin."""
import logging
import os
import re
from itertools import chain
from pathlib import Path

import hdbscan
import numpy as np
import vaex

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

POLUS_TAB_EXT = os.environ.get("POLUS_TAB_EXT", ".csv")
CHUNK_SIZE = 10000


def hdbscan_model(
    data: np.ndarray,
    min_cluster_size: int,
    increment_outlier_id: bool,
) -> np.ndarray:
    """Cluster data using HDBSCAN.

    Args:
        data: Data that need to be clustered.
        min_cluster_size: Minimum cluster size.
        increment_outlier_id : Increment outlier ID to unity.

    Returns:
        Cluster labels for each row of data.
    """
    clusters = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size).fit(data)
    labels = clusters.labels_.flatten().astype(np.uint16) + 1
    return labels + 1 if increment_outlier_id else labels


def hdbscan_clustering(  # noqa: PLR0913
    file: Path,
    min_cluster_size: int,
    out_dir: Path,
    grouping_pattern: str,
    label_col: str,
    average_groups: bool,
    increment_outlier_id: bool,
) -> None:
    """Cluster data using HDBSCAN.

    Args:
        file: Path of a tabular file.
        min_cluster_size: Smallest size grouping that should be considered as a cluster.
        out_dir: Path to output directory.
        grouping_pattern: Regular expression to caputure groups in a label_col.
        label_col: Name of column containing labels.
        average_groups:To average data across groups.
        increment_outlier_id: Increment outlier ID to unity.
    """
    if Path(file.name).suffix == ".csv":
        df = vaex.from_csv(file, convert=True, chunk_size=CHUNK_SIZE)
    else:
        df = vaex.open(file)
    # If user provided a regular expression.
    if grouping_pattern:
        if label_col == "None":
            msg = f"Please define label column to capture groups {label_col}"
            raise ValueError(msg)

        # Create a column group with matching string
        group = np.array(
            [
                re.search(grouping_pattern, x).group(0)  # type: ignore
                for x in df[label_col].tolist()
                if len(re.search(grouping_pattern, x).group(0)) != 0  # type: ignore
            ],
        )
        if len(group) == 0:
            msg = f"Could not find group with pattern {grouping_pattern}"
            raise ValueError(msg)

        # Create a column group with matching string
        df["group"] = group
        int_columns = [
            feature
            for feature in df.get_column_names()
            if df.data_type(feature) == int or df.data_type(feature) == float
        ]

        # If we want to average features for each group.
        if average_groups:
            df_grouped = df.groupby(
                "group",
                agg=[vaex.agg.mean(x) for x in int_columns],
            )
            # Cluster data using HDBSCAN clustering.
            logger.info("Clustering the data")
            cluster_ids = hdbscan_model(
                df_grouped.values,
                min_cluster_size,
                increment_outlier_id,
            )
            df_grouped["cluster"] = cluster_ids
            df = df.join(
                df_grouped["group", "cluster"],
                left_on="group",
                right_on="group",
            )

        else:
            dfs = []
            for group, df_ss in df.groupby("group"):
                # Cluster data using HDBSCAN clustering.
                logger.info(f"Clustering data in group {group}")

                cluster_ids = hdbscan_model(
                    df_ss.values,
                    min_cluster_size,
                    increment_outlier_id,
                )

                dfs.append(cluster_ids)
            cluster_ids = np.array(list(chain.from_iterable(dfs)))
            df["cluster"] = cluster_ids

    # No grouping. Vanilla clustering.
    else:
        int_columns = [
            feature
            for feature in df.get_column_names()
            if df.data_type(feature) == int or df.data_type(feature) == float
        ]

        # Cluster data using HDBSCAN clustering
        logger.info("Clustering the data")
        cluster_ids = hdbscan_model(
            df[int_columns].values,
            min_cluster_size,
            increment_outlier_id,
        )
        df["cluster"] = cluster_ids

    outname = Path(out_dir, f"{Path(file.name).stem}_hdbscan{POLUS_TAB_EXT}")

    if POLUS_TAB_EXT == ".arrow":
        df.export_feather(outname)
        logger.info(f"Saving outputs: {outname}")
    else:
        df.export_csv(path=outname, chunk_size=CHUNK_SIZE)

    logger.info("Finished all processes!")
