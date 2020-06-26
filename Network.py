import os
from shutil import copyfile
import pandas as pd
from pprint import pprint
import csv
from Node import Node, NodeType
from Position import Position
import numpy as np
import scipy.io

import matplotlib.pyplot as plt

inf = float("inf")


class Network:
    def __init__(self, folder_path, dimension):
        self.folder_path = folder_path
        self.dimension = dimension
        anchor_path = folder_path + "anchor.csv"
        parameter_path = folder_path + "parameters of network.txt"

        with open(parameter_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                line = line.replace(";", "")
                lhs, rhs = line.split("=", 2)
                lhs = lhs.strip()
                rhs = rhs.strip()
                if lhs == "radio range":
                    self.radio_range = float(rhs)
                elif lhs == "number of sensors":
                    self.number_of_sensors = int(rhs)
                elif lhs == "number of anchors":
                    self.number_of_anchors = int(rhs)

        self.sensors = [Node(NodeType.Sensor) for _ in range(self.number_of_sensors)]

        self.anchors = []
        #  Init anchors
        with open(anchor_path, 'r') as file:
            data = file.readlines()
            for i in range(len(data)):
                data[i] = data[i].strip()
                data[i] = data[i].split(",")

            assert len(data) == self.dimension
            for i in range(len(data[0])):
                x = float(data[0][i])
                y = float(data[1][i])
                if self.dimension == 3:
                    z = float(data[2][i])
                    anchor = Node(NodeType.Anchor, Position(x, y, z))
                    self.anchors.append(anchor)

                else:
                    anchor = Node(NodeType.Anchor, Position(x, y))
                    self.anchors.append(anchor)

        # init hop to anchors
        # for sensor in self.sensors:
        #     for i in range(len(self.anchors)):
        #         sensor.hop_to_anchor[i] = float('Inf')

        sensors_csv_path = folder_path + "netss.csv"
        data = list(csv.reader(open(sensors_csv_path)))
        assert len(data) == len(self.sensors)

        for i in range(len(data)):
            for j in range(len(data[i])):
                value = float(data[i][j])
                assert data[i][j] == data[j][i]
                if value != 0:
                    self.sensors[i].add_dist_to_neighbour(j, value)

        node_sensor_csv_path = folder_path + "netsa.csv"
        data = list(csv.reader(open(node_sensor_csv_path)))
        assert len(data) == len(self.anchors)
        assert len(data[0]) == len(self.sensors)

        for i in range(len(data)):
            for j in range(len(data[i])):
                value = float(data[i][j])
                if value != 0:
                    self.anchors[i].add_dist_to_neighbour(j, value)
                    self.sensors[j].add_dist_to_anchor(i, value)
                    self.sensors[j].hop_to_anchor[i] = 1
                    self.sensors[j].path_to_anchor[i] = [-1]
                else:
                    self.anchors[i].add_dist_to_neighbour(j, inf)
                    self.sensors[j].add_dist_to_anchor(i, inf)
                    self.sensors[j].hop_to_anchor[i] = inf
                    self.sensors[j].path_to_anchor[i] = []

        self.init_hop()
        self.init_hop_size()
        self.set_sensor_distance_to_anchor()

    def init_hop(self):
        #  For sensor to sensor
        do_continue = True
        while do_continue:
            do_continue = False
            for i in range(len(self.sensors)):
                for j in range(len(self.sensors)):
                    if i == j:
                        continue
                    if j not in self.sensors[i].distance_to_neighbour.keys():
                        assert i not in self.sensors[j].distance_to_neighbour.keys()
                        continue

                    for k in range(len(self.anchors)):
                        if self.sensors[i].hop_to_anchor[k] + 1 < self.sensors[j].hop_to_anchor[k]:
                            do_continue = True
                            self.sensors[j].hop_to_anchor[k] = self.sensors[i].hop_to_anchor[k] + 1
                            self.sensors[j].path_to_anchor[k] = [i] + [p for p in self.sensors[i].path_to_anchor[k]]
        for sensor in self.sensors:
            for i in range(len(self.anchors)):
                assert len(sensor.path_to_anchor[i]) == sensor.hop_to_anchor[i]

        #  For anchor to anchor
        for i in range(len(self.anchors)):
            for j in range(len(self.anchors)):
                if i == j:
                    self.anchors[i].hop_to_anchor[j] = 0
                    continue
                neighbours = []

                for k in self.anchors[i].distance_to_neighbour.keys():
                    if self.anchors[i].distance_to_neighbour[k] != inf and self.anchors[i].distance_to_neighbour[
                        k] != 0:
                        neighbours.append(self.sensors[k])
                self.anchors[i].hop_to_anchor[j] = min(neighbours, key=lambda x: x.hop_to_anchor[j]).hop_to_anchor[
                                                       j] + 1

    def init_hop_size(self):
        for i in range(len(self.anchors)):
            sum = 0
            sum_hop_count = 0
            for j in range(len(self.anchors)):
                if i == j:
                    continue
                sum += self.anchors[i].distance_to(self.anchors[j])
                sum_hop_count += self.anchors[i].hop_to_anchor[j]
            self.anchors[i].hop_size = sum / sum_hop_count

    def set_sensor_distance_to_anchor(self):
        for sensor in self.sensors:
            for i in range(len(self.anchors)):
                if sensor.distance_to_anchor[i] != inf:
                    continue
                assert self.anchors[i].hop_size != inf
                sensor.distance_to_anchor[i] = sensor.hop_to_anchor[i] * self.anchors[i].hop_size

    def calculate_distance(self):
        n = len(self.anchors) - 1
        for sensor in self.sensors:
            #  Construct A
            A = np.ndarray([n, 2])
            for i in range(len(A)):
                A[i][0] = self.anchors[i].position.x - self.anchors[n].position.x
                A[i][1] = self.anchors[i].position.y - self.anchors[n].position.y
            A = -2 * A
            B = np.ndarray([n, 1])
            for i in range(len(B)):
                assert i != n
                B[i][0] = sensor.distance_to_anchor[i] ** 2 - sensor.distance_to_anchor[n] ** 2
                B[i][0] += -(self.anchors[i].position.x ** 2) + (self.anchors[n].position.x ** 2)
                B[i][0] += -(self.anchors[i].position.y ** 2) + (self.anchors[n].position.y ** 2)

            X = np.dot(np.linalg.inv(np.dot(A.transpose(), A)), A.transpose())
            X = np.dot(X, B)

            sensor.position = Position(X[0][0], X[1][0])
        pass

    def plot(self):
        x = [sensor.position.x for sensor in self.sensors]
        y = [sensor.position.y for sensor in self.sensors]
        plt.scatter(x, y, color='r', marker='+', label="sensors")
        x = [anchor.position.x for anchor in self.anchors]
        y = [anchor.position.y for anchor in self.anchors]
        plt.scatter(x, y, label="anchors", color='b', marker='x')
        plt.legend()
        plt.show()

    def get_sensor_nerar(self, x, y, offset=0.1):
        result = dict()
        for i in range(len(self.sensors)):
            if abs(self.sensors[i].position.x - x) < offset and abs(self.sensors[i].position.y - y) < offset:
                result[i] = self.sensors[i]
        return result

    def generate_random_tests(self, sensor_count=None):
        if sensor_count is None:
            sensor_count = len(self.sensors)
        generated_sensors = []
        for i in range(sensor_count):
            x = np.random.uniform(-0.5, 0.5)
            y = np.random.uniform(-0.5, 0.5)
            sensor = Node(NodeType.Sensor, actual_position=Position(x, y))
            generated_sensors.append(sensor)
        i = 1
        while os.path.exists(self.folder_path + "test%s/" % i):
            i += 1
        test_folder = self.folder_path + "test%s/" % i
        os.makedirs(test_folder)

        sensor_to_sensor = test_folder + "netss.csv"
        with open(sensor_to_sensor, 'w') as file:
            for i in range(sensor_count):
                distances = []
                for j in range(sensor_count):
                    dis = generated_sensors[i].actual_position.distance_to(generated_sensors[j].actual_position)
                    if dis > self.radio_range:
                        dis = 0
                    distances.append(str(dis))
                file.write(",".join(distances))
                file.write("\n")
        anchor_to_sensor = test_folder + "netsa.csv"
        with open(anchor_to_sensor, 'w') as file:
            for i in range(len(self.anchors)):
                distances = []
                for j in range(sensor_count):
                    dis = self.anchors[i].position.distance_to(generated_sensors[j].actual_position)
                    if dis > self.radio_range:
                        dis = 0
                    distances.append(str(dis))
                file.write(",".join(distances))
                file.write("\n")

        copyfile(self.folder_path + "anchor.csv", test_folder + "anchor.csv")
        copyfile(self.folder_path + "parameters of network.txt", test_folder + "parameters of network.txt")

        actual_positions = test_folder + "actual_position.csv"
        with open(actual_positions, 'w') as file:
            data = []
            for i in range(sensor_count):
                data.append(str(generated_sensors[i].actual_position.x))
            file.write(",".join(data))
            file.write("\n")

            data = []
            for i in range(sensor_count):
                data.append(str(generated_sensors[i].actual_position.y))
            file.write(",".join(data))
            file.write("\n")

            if self.dimension == 3:
                data = []
                for i in range(sensor_count):
                    data.append(str(generated_sensors[i].actual_position.z))
                file.write(",".join(data))
                file.write("\n")

        return TestNetwork(test_folder, self.dimension)


class TestNetwork(Network):
    def __init__(self, folder_path, dimension):
        super().__init__(folder_path, dimension)
        self.load_actual_position()

    def load_actual_position(self):
        actual_position_path = self.folder_path + "actual_position.csv"
        with open(actual_position_path, 'r') as file:
            data = file.readlines()
            for i in range(len(data)):
                data[i] = data[i].strip()
                data[i] = data[i].split(",")

            assert len(data) == self.dimension
            for i in range(len(data[0])):
                x = float(data[0][i])
                y = float(data[1][i])
                if self.dimension == 3:
                    z = float(data[2][i])
                    self.sensors[i].actual_position = Position(x, y, z)
                else:
                    self.sensors[i].actual_position = Position(x, y)

    def calculate_error(self):
        self.calculate_distance()
        error_vector = []
        for sensor in self.sensors:
            error_vector.append(sensor.position.distance_to(sensor.actual_position))

        assert len(error_vector) == len(self.sensors)
        return error_vector

    def plot(self):
        x = [sensor.position.x for sensor in self.sensors]
        y = [sensor.position.y for sensor in self.sensors]
        plt.scatter(x, y, color='r', marker='+', label="sensors")
        x = [sensor.actual_position.x for sensor in self.sensors]
        y = [sensor.actual_position.y for sensor in self.sensors]
        plt.scatter(x, y, color='g', marker='+', label="actual")
        x = [anchor.position.x for anchor in self.anchors]
        y = [anchor.position.y for anchor in self.anchors]
        plt.scatter(x, y, label="anchors", color='b', marker='x')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    network = Network("./data-csv/data1/", 2)

    test_network = network.generate_random_tests()

    errors = test_network.calculate_error()
    test_network.plot()
    average_error = np.mean(errors)
    print(average_error)
