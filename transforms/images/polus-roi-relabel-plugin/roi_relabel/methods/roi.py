import math
import typing


class Point(typing.NamedTuple):

    y: float
    x: float

    def __lt__(self, other: 'Point') -> bool:
        return self.x < other.x or self.y < other.y

    def __add__(self, other: 'Point') -> 'Point':
        return Point(self.x + other.x, self.y + other.y)

    def min(self, other: 'Point') -> 'Point':
        return Point(
            x=min(self.x, other.x),
            y=min(self.y, other.y),
        )

    def max(self, other: 'Point') -> 'Point':
        return Point(
            x=max(self.x, other.x),
            y=max(self.y, other.y),
        )

    def mid(self, other: 'Point') -> 'Point':
        return Point(
            x=(self.x + other.x) / 2,
            y=(self.y + other.y) / 2,
        )

    def distance_to(self, other: 'Point') -> float:
        return math.hypot(self.x - other.x, self.y - other.y)


class RoI:

    __slots__ = [
        'start',
        'end',
        'label',
    ]

    def __init__(self, start: tuple[int, int], end: tuple[int, int], label: int):
        self.start = Point(*start)
        self.end = Point(*end)
        self.label = label

    def __lt__(self, other: 'RoI') -> bool:
        if self.start == other.start:
            return self.end < other.end
        else:
            return self.start < other.start

    def __hash__(self) -> int:
        return hash(self.label)

    @property
    def center(self) -> 'Point':
        return self.start.mid(self.end)

    @property
    def range(self) -> float:
        return self.center.distance_to(self.start)

    def merge_with(self, other: 'RoI') -> 'RoI':
        return RoI(
            start=self.start.min(other.start),
            end=self.end.max(other.end),
            label=self.label
        )

    def touches(self, other: 'RoI') -> bool:
        if other < self:
            return other.touches(self)
        else:
            return self.end.x >= other.start.x and self.end.y >= other.start.y

    def in_range_of(self, other: 'RoI', multiplier: float = 1.0) -> bool:
        d = self.center.distance_to(other.center)
        return d <= ((self.range + other.range) * multiplier)
