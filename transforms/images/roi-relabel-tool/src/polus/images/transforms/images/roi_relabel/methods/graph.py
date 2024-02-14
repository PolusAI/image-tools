"""Helpful Graphs for using networkx."""

import networkx

from . import roi


class Graph:
    """Helpful Graphs for using networkx."""

    def __init__(self, rois: list[roi.RoI], range_multiplier: float = 1.0) -> None:
        """Create a Graph from RoIs.

        Args:
            rois: A list of RoIs.
            range_multiplier: How far to look for neighbors.
        """
        self.graph = networkx.Graph()
        self.graph.add_nodes_from(rois)

        self.graph.add_edges_from(
            (
                (u, v)
                for i, u in enumerate(rois)
                for v in rois[i + 1 :]
                if u.in_range_of(v, multiplier=range_multiplier)
            ),
        )

    def coloring(self, max_val: int, optimize: bool = False) -> dict[roi.RoI, int]:
        """Get integer labels for RoIs based on a graph coloring.

        Args:
            max_val: spreads out labels evenly in the [1, max_val] range.
            optimize: Optimize graph coloring to make nearby RoIs have maximally
             different labels (TODO).

        Returns:
            A mapping of RoIs to integer labels.
        """
        colors: dict[roi.RoI, int] = networkx.coloring.greedy_color(self.graph)

        if optimize:
            # TODO
            raise NotImplementedError

        num_colors = len(set(colors.values()))
        step = (max_val - 2) // num_colors
        return {k: 1 + (v * step) for k, v in colors.items()}
