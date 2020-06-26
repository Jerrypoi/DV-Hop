from math import sqrt


class Position:
    def __init__(self, x, y, z=None):
        self.x = x
        self.y = y
        self.z = z


    def __eq__(self, other):
        if not isinstance(other, Position):
            return False
        return self.x == other.x and self.y == other.y and self.z == other.z

    def distance_to(self, rhs) -> float:
        if not isinstance(rhs, Position):
            raise ValueError(str(rhs) + " is not a valid position")
        if self.z is None:
            return sqrt((self.x - rhs.x) * (self.x - rhs.x) + (self.y - rhs.y) * (self.y - rhs.y))
        else:
            return sqrt((self.x - rhs.x) * (self.x - rhs.x) + (self.y - rhs.y) * (self.y - rhs.y) + (self.z - rhs.z) * (self.z - rhs.z))

    def __str__(self):
        if self.z is None:
            return "(" + ", ".join([str(self.x), str(self.y)]) + ")"
        else:
            return "(" + ", ".join([str(self.x), str(self.y), str(self.z)]) + ")"
