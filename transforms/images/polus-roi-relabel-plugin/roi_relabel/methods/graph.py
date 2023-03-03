import networkx

from roi_relabel.utils import helpers
from . import roi

logger = helpers.make_logger(__name__)


class Graph:
    def __init__(self, rois: list[roi.RoI], range_multiplier: float = 1.0):
        self.graph = networkx.Graph()

        self.graph.add_edges_from((
            (u, v)
            for i, u in enumerate(rois)
            for v in rois[i + 1:]
            if u.in_range_of(v, multiplier=range_multiplier)
        ))

    def coloring(self, max_val: int, optimize: bool = False) -> dict[roi.RoI, int]:
        colors: dict[roi.RoI, int] = networkx.coloring.greedy_color(self.graph)

        if optimize:
            # TODO
            raise NotImplementedError

        num_colors = len(set(colors.values()))
        step = (max_val - 2) // (num_colors + 1)
        colors = {k: (v + 1) * step for k, v in colors.items()}

        return colors
