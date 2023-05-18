"""K_means clustering."""
import logging
import pathlib

import numpy
import numpy as np
import numpy.matlib
import vaex
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score

from polus.plugins.clustering.k_means.utils import Extensions, Methods

# Initialize the logger
logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger("main")
logger.setLevel(logging.INFO)


def elbow(data_array: np.array, minimum_range: int, maximum_range: int) -> np.array:
    """Determine k value and cluster data using elbow method.

    Args:
        data_array : Input data.
        minimum_range : Starting number of sequence in range function to determine k-value.
        maximum_range : Ending number of sequence in range function to determine k-value.

    Returns:
        Labeled data.
    """
    sse = []
    label_value = []
    logger.info("Starting Elbow Method...")
    K = range(minimum_range, maximum_range + 1)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=9).fit(data_array)
        centroids = kmeans.cluster_centers_
        pred_clusters = kmeans.predict(data_array)
        curr_sse = 0

        # calculate square of Euclidean distance of each point from its cluster center and add to current WSS
        logger.info("Calculating Euclidean distance...")
        for i in range(len(data_array)):
            curr_center = centroids[pred_clusters[i]]
            curr_sse += np.linalg.norm(data_array[i] - np.array(curr_center)) ** 2
        sse.append(curr_sse)
        labels = kmeans.labels_
        label_value.append(labels)

    logger.info("Finding elbow point in curve...")
    # Find the elbow point in the curve
    points = len(sse)
    # Get coordinates of all points
    coord = np.vstack((range(points), sse)).T
    # First point
    f_point = coord[0]
    # Vector between first and last point
    linevec = coord[-1] - f_point
    # Normalize the line vector
    linevecn = linevec / np.sqrt(np.sum(linevec**2))
    # Vector between all point and first point
    vecf = coord - f_point
    # Parallel vector
    prod = np.sum(vecf * numpy.matlib.repmat(linevecn, points, 1), axis=1)
    vecfpara = np.outer(prod, linevecn)
    # Perpendicular vector
    vecline = vecf - vecfpara
    # Distance from curve to line
    dist = np.sqrt(np.sum(vecline**2, axis=1))
    # Maximum distance point
    k_cluster = np.argmax(dist) + minimum_range
    logger.info("k cluster: %s", k_cluster)
    logger.info("label value: %s", label_value)
    logger.info("Setting label_data")
    label_data = label_value[k_cluster]
    return label_data


def calinski_davies(
    data_array: np.array, methods: Methods, minimum_range: int, maximum_range: int
) -> np.array:
    """Determine k value and cluster data using Calinski Harabasz Index method or Davies Bouldin based on method selection.

    Args:
        data: Input data.
        methods: Select either Calinski Harabasz or Davies Bouldin method.
        minimum_range: Starting number of sequence in range function to determine k-value.
        maximum_range:Ending number of sequence in range function to determine k-value.

    Returns:
        Labeled data.
    """
    K = range(minimum_range, maximum_range + 1)
    chdb = []
    label_value = []
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=9).fit(data_array)
        labels = kmeans.labels_
        label_value.append(labels)
        if f"{methods}" == "CalinskiHarabasz":
            ch_db = calinski_harabasz_score(data_array, labels)
        else:
            ch_db = davies_bouldin_score(data_array, labels)
        chdb.append(ch_db)
    if f"{methods}" == "CalinskiHarabasz":
        score = max(chdb)
    else:
        score = min(chdb)
    k_cluster = chdb.index(score)
    label_data = label_value[k_cluster]
    return label_data


# Methods_DICT = {
#     Methods.ELBOW: elbow,
#     Methods.CALINSKIHARABASZ: calinski_davies,
#     Methods.DAVIESBOULDIN: calinski_davies,
#     Methods.Default: elbow,
# }


def clustering(
    file: pathlib.Path,
    file_pattern: str,
    methods: Methods,
    minimum_range: int,
    maximum_range: int,
    num_of_clus: int,
    file_extension: Extensions,
    out_dir: pathlib.Path,
):
    """K-means clustering methods to find clusters of similar or more related objects.

    Args:
        file: Input path.
        file_pattern: Pattern to parse tabular files.
        methods: Select either Calinski Harabasz or Davies Bouldin method or Manual.
        minimum_range: Starting number of sequence in range function to determine k-value.
        maximum_range:Ending number of sequence in range function to determine k-value.
        file_extension: Output file format
    """
    # Get file name
    filename = file.stem
    logger.info("Started reading the file " + file.name)
    with open(file, encoding="utf-8", errors="ignore") as fr:
        ncols = len(fr.readline().split(","))
    chunk_size = max([2**24 // ncols, 1])
    if f"{file_pattern}" == ".csv":
        df = vaex.read_csv(file, convert=True, chunk_size=chunk_size)
    else:
        df = vaex.open(file)
    # Get list of column names
    cols = df.get_column_names()

    # Separate data by categorical and numerical data types
    numerical = []
    categorical = []
    for col in cols:
        if df[col].dtype == str:
            categorical.append(col)
        else:
            numerical.append(col)
    # Remove label field
    if "label" in numerical:
        numerical.remove("label")

    if numerical is None:
        raise ValueError("There are no numerical features in the data.")
    else:
        data = df[numerical]

    if categorical:
        cat_array = df[categorical]
    else:
        logger.info("No categorical features found in the data")

    if f"{methods}" != "Manual":
        # Check whether minimum range and maximum range value is entered
        if methods and not (minimum_range or maximum_range):
            raise ValueError(
                "Enter both minimumrange and maximumrange to determine k-value."
            )
        if minimum_range <= 1:
            raise ValueError("Minimumrange should be greater than 1.")
        logger.info(
            "Determining k-value using " + methods + " and clustering the data."
        )
        if f"{methods}" == "CalinskiHarabasz":
            label_data = calinski_davies(data, methods, minimum_range, maximum_range)
        if f"{methods}" == "DaviesBouldin":
            label_data = calinski_davies(data, methods, minimum_range, maximum_range)
        if f"{methods}" == "Elbow":
            label_data = elbow(data, minimum_range, maximum_range)
    else:
        # Check whether numofclus is entered
        if not num_of_clus:
            raise ValueError("Enter number of clusters")
        kvalue = num_of_clus
        kmeans = KMeans(n_clusters=kvalue).fit(data)
        label_data = kmeans.labels_

    # Cluster data using K-Means clustering
    logger.info("Adding Cluster Data")
    data["Cluster"] = label_data

    # Add Categorical Data back to data processed
    if categorical:
        logger.info("Adding categorical data")
        for col in categorical:
            data[col] = cat_array[col].values

    # Save dataframe to feather file or to csv file
    out_file = pathlib.Path(out_dir, (filename + file_extension))

    if f"{file_extension}" in [".feather", ".arrow"]:
        data.export_feather(out_file)
    elif f"{file_extension}" == ".parquet":
        data.export_parquet(out_file)
    elif f"{file_extension}" == ".hdf5":
        data.export_hdf5(out_file)
    else:
        logger.info("Saving csv file")
        data.export_csv(out_file, chunk_size=chunk_size)
