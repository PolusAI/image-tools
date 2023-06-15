"""K_means clustering."""
import enum


class Methods(str, enum.Enum):
    """Clustering methods to determine k-value."""

    ELBOW = "Elbow"
    CALINSKIHARABASZ = "CalinskiHarabasz"
    DAVIESBOULDIN = "DaviesBouldin"
    MANUAL = "Manual"
    Default = "Elbow"
