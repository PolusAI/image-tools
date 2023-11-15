"""RoI objects."""

import math

__all__ = [
    "RoI",
]


class Point:
    """A point in 2D space. Used to represent the corners of a rectangle."""

    def __init__(self, x: int, y: int) -> None:
        """Create a new point."""
        self.x = x
        self.y = y

    def __str__(self) -> str:
        """Return a string representation of the point."""
        return f"(x = {self.x:3d}, y = {self.y:3d})"

    def __repr__(self) -> str:
        """Return a string representation of the point."""
        return str(self)

    def __lt__(self, other: "Point") -> bool:
        """Compare and order points."""
        return self.x < other.x and self.y < other.y

    def __add__(self, other: "Point") -> "Point":
        """Add two points together."""
        return Point(self.x + other.x, self.y + other.y)

    def min_(self, other: "Point") -> "Point":
        """Return the coordinate-wise minimum of two points."""
        return Point(
            x=min(self.x, other.x),
            y=min(self.y, other.y),
        )

    def max_(self, other: "Point") -> "Point":
        """Return the coordinate-wise maximum of two points."""
        return Point(
            x=max(self.x, other.x),
            y=max(self.y, other.y),
        )

    def mid_(self, other: "Point") -> "Point":
        """Return the midpoint of two points."""
        return Point(
            x=(self.x + other.x) // 2,
            y=(self.y + other.y) // 2,
        )

    def distance_to(self, other: "Point") -> float:
        """Compute the distance between two points."""
        x = self.x - other.x
        y = self.y - other.y
        return math.sqrt(x * x + y * y)

    def as_tuple(self) -> tuple[int, int]:
        """Return the coordinates of the point as a tuple."""
        return self.x, self.y


class RoI:
    """Create and manage RoIs from a labeled image."""

    def __init__(
        self,
        top_left: tuple[int, int],
        bottom_right: tuple[int, int],
        label: int,
    ) -> None:
        """Create a new RoI from the top-left and bottom-right corners of a rectangle.

        Args:
            top_left: corner of rectangle.
            bottom_right: corner of rectangle.
            label: integer label for RoI. Also used for hashing RoIs for use
             with graph-based algorithms.
        """
        self.top_left = Point(*top_left)
        self.bottom_right = Point(*bottom_right)

        if not (self.top_left < self.bottom_right):
            msg = "RoI must have positive area."
            raise ValueError(msg)

        self.label = label
        self.center = self.top_left.mid_(self.bottom_right)

    def __str__(self) -> str:
        """Return a string representation of the RoI."""
        return (
            f"RoI(top-left = {self.top_left}, bottom-right = {self.bottom_right}, "
            f"label = {self.label:3d})"
        )

    def __repr__(self) -> str:
        """Return a string representation of the RoI."""
        return str(self)

    def __lt__(self, other: "RoI") -> bool:
        """Compare and order RoIs.

        Args:
            other: RoI for comparison.

        Returns:
            Whether `self` should be ordered before `other`.
        """
        if self.top_left == other.top_left:
            if self.bottom_right.x == other.bottom_right.x:
                return self.bottom_right.y < other.bottom_right.y

            return self.bottom_right.x < other.bottom_right.x

        if self.top_left.x == other.top_left.x:
            return self.top_left.y < other.top_left.y

        return self.top_left.x < other.top_left.x

    def __eq__(self, other: "RoI") -> bool:  # type: ignore[override]
        """Compare RoIs for equality of the bounding boxes and the label.

        Args:
            other: RoI for comparison.

        Returns:
            Whether `self` is equal to `other`.
        """
        return all(
            (
                self.top_left == other.top_left,
                self.bottom_right == other.bottom_right,
                self.label == other.label,
            ),
        )

    def __ne__(self, other: "RoI") -> bool:  # type: ignore[override]
        """Compare RoIs for inequality of the bounding boxes.

        Args:
            other: RoI for comparison.

        Returns:
            Whether `self` is not equal to `other`.
        """
        return not (self == other)

    def __gt__(self, other: "RoI") -> bool:
        """Compare and order RoIs.

        Args:
            other: RoI for comparison.

        Returns:
            Whether `self` should be ordered after `other`.
        """
        return other < self

    def __le__(self, other: "RoI") -> bool:
        """Compare and order RoIs.

        Args:
            other: RoI for comparison.

        Returns:
            Whether `self` should be ordered before or equal to `other`.
        """
        return self < other or self == other

    def __ge__(self, other: "RoI") -> bool:
        """Compare and order RoIs.

        Args:
            other: RoI for comparison.

        Returns:
            Whether `self` should be ordered after or equal to `other`.
        """
        return other < self or other == self

    def __hash__(self) -> int:
        """Hash RoIs for use in networkx graphs."""
        return hash(self.label)

    @property
    def range_(self) -> float:
        """Compute the radial range of the RoI."""
        return self.center.distance_to(self.top_left)

    def merge_with(self, other: "RoI") -> "RoI":
        """Merge with another RoI and return the new ROI.

        Args:
            other: an RoI.
        """
        return RoI(
            top_left=self.top_left.min_(other.top_left).as_tuple(),
            bottom_right=self.bottom_right.max_(other.bottom_right).as_tuple(),
            label=self.label,
        )

    def touches(self, other: "RoI") -> bool:
        """Check whether the bounding boxes of the RoIs touch each other.

        Args:
            other: an RoI.
        """
        if other < self:
            return other.touches(self)

        return all(
            (
                self.top_left.x <= other.top_left.x <= self.bottom_right.x,
                self.top_left.y <= other.top_left.y <= self.bottom_right.y,
            ),
        )

    def in_range_of(self, other: "RoI", multiplier: float = 1.0) -> bool:
        """Check whether the two RoIs within range to have different labels.

        Args:
            other: an RoI.
            multiplier: on the default radial range.
        """
        d = self.center.distance_to(other.center)
        return d <= ((self.range_ + other.range_) * multiplier)
