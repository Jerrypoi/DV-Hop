from enum import Enum

from Position import Position

inf = float("inf")


class NodeType(Enum):
    Anchor = 1
    Sensor = 2


class Node:
    def __init__(self, node_type: NodeType, position: Position = None, actual_position: Position = None):
        self.node_type = node_type
        self.position = position
        self.distance_to_neighbour = dict()
        self.distance_to_anchor = dict()
        self.hop_to_anchor = dict()
        self.path_to_anchor = dict()
        self.hop_size = inf
        self.actual_position = actual_position

    def __str__(self) -> str:
        return str(self.node_type) + ", " + str(self.position)

    def __repr__(self):
        return self.__str__()

    def add_dist_to_neighbour(self, neighbour_index: int, distance: float):
        self.distance_to_neighbour[neighbour_index] = distance

    def add_dist_to_anchor(self, anchor_index: int, distance: float):
        self.distance_to_anchor[anchor_index] = distance

    def distance_to(self, rhs) -> float:
        if not isinstance(rhs, Node):
            raise ValueError(str(rhs) + " is not a node!")
        if rhs.position is None:
            raise ValueError(str(rhs) + " do not have a position")

        return self.position.distance_to(rhs.position)
