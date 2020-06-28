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
                    self.sensors[j].neighbouring_anchor += 1
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

    def plot(self):
        x = [sensor.position.x for sensor in self.sensors]
        y = [sensor.position.y for sensor in self.sensors]
        plt.scatter(x, y, color='r', marker='+', label="sensor")
        x = [anchor.position.x for anchor in self.anchors]
        y = [anchor.position.y for anchor in self.anchors]
        plt.scatter(x, y, label="anchor", color='b', marker='x')
        plt.legend()
        plt.savefig(self.folder_path + "dv_hop.png", dpi=600)
        plt.show()


    def get_sensor_near(self, x, y, offset=0.1):
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
        with open(test_folder + "parameters of network.txt", 'w') as file:
            file.write("radion range={};\n".format(self.radio_range))
            file.write("number of sensors={};\n".format(sensor_count))
            file.write("number of anchors={};\n".format(len(self.anchors)))

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

    def plot_with_path(self):
        x = [sensor.position.x for sensor in self.sensors]
        y = [sensor.position.y for sensor in self.sensors]
        plt.scatter(x, y, color='r', marker='+', label="sensor")
        x = [anchor.position.x for anchor in self.anchors]
        y = [anchor.position.y for anchor in self.anchors]
        plt.scatter(x, y, label="anchor", color='b', marker='x')
        for sensor in self.sensors:
            for i in sensor.distance_to_neighbour.keys():
                if sensor.distance_to_neighbour[i] != inf and sensor.distance_to_neighbour[i] != 0:
                    x, y = [sensor.position.x, self.sensors[i].position.x], [sensor.position.y,
                                                                             self.sensors[i].position.y]
                    plt.plot(x, y, marker='o')
        plt.legend()
        plt.savefig("path.png", dpi=600)
        plt.show()


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
        plt.scatter(x, y, color='r', marker='+', label="sensor")
        x = [sensor.actual_position.x for sensor in self.sensors]
        y = [sensor.actual_position.y for sensor in self.sensors]
        plt.scatter(x, y, color='g', marker='+', label="actual")
        x = [anchor.position.x for anchor in self.anchors]
        y = [anchor.position.y for anchor in self.anchors]
        plt.scatter(x, y, label="anchor", color='b', marker='x')
        plt.legend()
        plt.show()


class FailedImprovedNetwork(Network):
    def calculate_distance(self):
        done_calculated = set()
        neighbouring_anchor_count = len(self.anchors)
        n = len(self.anchors) - 1
        use_neighbouring_anchor = -1
        while neighbouring_anchor_count >= 0:
            for index in range(len(self.sensors)):
                if index in done_calculated or self.sensors[index].neighbouring_anchor < neighbouring_anchor_count:
                    continue
                assert self.sensors[index].neighbouring_anchor == neighbouring_anchor_count
                #  Construct A
                if use_neighbouring_anchor == -1:
                    use_neighbouring_anchor = neighbouring_anchor_count
                additional_using_anchors = set()
                for calculated_index in done_calculated:
                    if index in self.sensors[calculated_index].distance_to_neighbour.keys():
                        if self.sensors[calculated_index].distance_to_neighbour[index] != inf and \
                                self.sensors[calculated_index].distance_to_neighbour[index] != 0 and \
                                self.sensors[calculated_index].neighbouring_anchor == use_neighbouring_anchor and \
                                neighbouring_anchor_count != use_neighbouring_anchor:
                            additional_using_anchors.add(calculated_index)
                additional_n = len(additional_using_anchors)
                A = np.ndarray([n + additional_n, 2])
                i = 0

                while i < n:
                    A[i][0] = self.anchors[i].position.x - self.anchors[n].position.x
                    A[i][1] = self.anchors[i].position.y - self.anchors[n].position.y
                    i += 1
                for done_index in additional_using_anchors:
                    A[i][0] = self.sensors[done_index].position.x - self.anchors[n].position.x
                    A[i][1] = self.sensors[done_index].position.y - self.anchors[n].position.y
                    i += 1
                assert i == n + additional_n
                A = -2 * A
                B = np.ndarray([n + additional_n, 1])
                i = 0
                while i < n:
                    B[i][0] = self.sensors[index].distance_to_anchor[i] ** 2 - self.sensors[index].distance_to_anchor[
                        n] ** 2
                    B[i][0] += -(self.anchors[i].position.x ** 2) + (self.anchors[n].position.x ** 2)
                    B[i][0] += -(self.anchors[i].position.y ** 2) + (self.anchors[n].position.y ** 2)
                    i += 1

                for done_index in additional_using_anchors:
                    B[i][0] = self.sensors[index].distance_to_neighbour[done_index] ** 2 - \
                              self.sensors[i].distance_to_anchor[
                                  n] ** 2
                    B[i][0] += -(self.sensors[done_index].position.x ** 2) + (self.anchors[n].position.x ** 2)
                    B[i][0] += -(self.sensors[done_index].position.y ** 2) + (self.anchors[n].position.y ** 2)
                    i += 1

                X = np.dot(np.linalg.inv(np.dot(A.transpose(), A)), A.transpose())
                X = np.dot(X, B)
                self.sensors[index].position = Position(X[0][0], X[1][0])
                done_calculated.add(index)
            neighbouring_anchor_count -= 1
        assert len(done_calculated) == len(self.sensors)
        # n = len(self.anchors) - 1
        # for sensor in self.sensors:
        #     #  Construct A
        #     A = np.ndarray([n, 2])
        #     for i in range(len(A)):
        #         A[i][0] = self.anchors[i].position.x - self.anchors[n].position.x
        #         A[i][1] = self.anchors[i].position.y - self.anchors[n].position.y
        #     A = -2 * A
        #     B = np.ndarray([n, 1])
        #     for i in range(len(B)):
        #         assert i != n
        #         B[i][0] = sensor.distance_to_anchor[i] ** 2 - sensor.distance_to_anchor[n] ** 2
        #         B[i][0] += -(self.anchors[i].position.x ** 2) + (self.anchors[n].position.x ** 2)
        #         B[i][0] += -(self.anchors[i].position.y ** 2) + (self.anchors[n].position.y ** 2)
        #
        #     X = np.dot(np.linalg.inv(np.dot(A.transpose(), A)), A.transpose())
        #     X = np.dot(X, B)
        #
        #     sensor.position = Position(X[0][0], X[1][0])
        # pass


class ImprovedNetworkWeighted(Network):
    def set_sensor_distance_to_anchor(self):
        for sensor in self.sensors:
            hop_size = 0
            hop_sum = sum([1 / sensor.hop_to_anchor[anchor_index] for anchor_index in range(len(self.anchors))])
            for i in range(len(self.anchors)):
                assert self.anchors[i].hop_size != inf
                weight = 1 / sensor.hop_to_anchor[i]
                weight /= hop_sum
                hop_size += self.anchors[i].hop_size * weight

            for i in range(len(self.anchors)):
                if sensor.distance_to_anchor[i] != inf:
                    continue
                assert self.anchors[i].hop_size != inf
                weight = 1 / sensor.hop_to_anchor[i]
                hop_sum = sum([1 / sensor.hop_to_anchor[anchor_index] for anchor_index in range(len(self.anchors))])
                weight /= hop_sum
                sensor.distance_to_anchor[i] = sensor.hop_to_anchor[i] * hop_size


class NewImprovedNetworkCalculateSomePointsFirst(ImprovedNetworkWeighted):
    def calculate_distance(self):
        done_calculated = set()
        for iter in range(len(self.sensors)):
            #  Construct A
            sensor = self.sensors[iter]
            if sensor.neighbouring_anchor >= 3:
                using_anchors = [self.anchors[temp_i] for temp_i in range(len(self.anchors)) if sensor.hop_to_anchor[temp_i] == 1]
                distances = [sensor.distance_to_anchor[temp_i] for temp_i in range(len(self.anchors)) if sensor.hop_to_anchor[temp_i] == 1]

                using_anchors += [self.sensors[temp_i] for temp_i in done_calculated if temp_i in sensor.distance_to_neighbour.keys() and sensor.distance_to_neighbour[temp_i] != inf and sensor.distance_to_neighbour[temp_i] != 0]
                distances += [sensor.distance_to_neighbour[temp_i] for temp_i in done_calculated if temp_i in sensor.distance_to_neighbour.keys() and sensor.distance_to_neighbour[temp_i] != inf and sensor.distance_to_neighbour[temp_i] != 0]

                assert len(using_anchors) == sensor.neighbouring_anchor
                assert len(distances) == sensor.neighbouring_anchor
                n = len(using_anchors) - 1

                A = np.ndarray([n, 2])

                for i in range(len(A)):
                    A[i][0] = using_anchors[i].position.x - using_anchors[n].position.x
                    A[i][1] = using_anchors[i].position.y - using_anchors[n].position.y
                A = -2 * A
                B = np.ndarray([n, 1])
                for i in range(len(B)):
                    assert i != n
                    B[i][0] = distances[i] ** 2 - distances[n] ** 2
                    B[i][0] += -(using_anchors[i].position.x ** 2) + (using_anchors[n].position.x ** 2)
                    B[i][0] += -(using_anchors[i].position.y ** 2) + (using_anchors[n].position.y ** 2)

                X = np.dot(np.linalg.inv(np.dot(A.transpose(), A)), A.transpose())
                X = np.dot(X, B)

                sensor.position = Position(X[0][0], X[1][0])
                done_calculated.add(iter)
                for index in range(len(self.sensors)):
                    if index == iter:
                        continue
                    if index in sensor.distance_to_neighbour.keys() and self.sensors[index].distance_to_neighbour[iter] != inf and self.sensors[index].distance_to_neighbour[iter] != 0:
                        self.sensors[index].neighbouring_anchor += 1
        n = len(self.anchors) - 1
        for index in range(len(self.sensors)):
            if index in done_calculated:
                continue

            additional_using_anchors = set()
            for calculated_index in done_calculated:
                if index in self.sensors[calculated_index].distance_to_neighbour.keys():
                    if self.sensors[calculated_index].distance_to_neighbour[index] != inf and \
                            self.sensors[calculated_index].distance_to_neighbour[index] != 0:
                        additional_using_anchors.add(calculated_index)
            additional_n = len(additional_using_anchors)
            A = np.ndarray([n + additional_n, 2])
            i = 0

            while i < n:
                A[i][0] = self.anchors[i].position.x - self.anchors[n].position.x
                A[i][1] = self.anchors[i].position.y - self.anchors[n].position.y
                i += 1
            for done_index in additional_using_anchors:
                A[i][0] = self.sensors[done_index].position.x - self.anchors[n].position.x
                A[i][1] = self.sensors[done_index].position.y - self.anchors[n].position.y
                i += 1
            assert i == n + additional_n
            A = -2 * A
            B = np.ndarray([n + additional_n, 1])
            i = 0
            while i < n:
                B[i][0] = self.sensors[index].distance_to_anchor[i] ** 2 - self.sensors[index].distance_to_anchor[
                    n] ** 2
                B[i][0] += -(self.anchors[i].position.x ** 2) + (self.anchors[n].position.x ** 2)
                B[i][0] += -(self.anchors[i].position.y ** 2) + (self.anchors[n].position.y ** 2)
                i += 1

            for done_index in additional_using_anchors:
                B[i][0] = self.sensors[index].distance_to_neighbour[done_index] ** 2 - \
                          self.sensors[i].distance_to_anchor[
                              n] ** 2
                B[i][0] += -(self.sensors[done_index].position.x ** 2) + (self.anchors[n].position.x ** 2)
                B[i][0] += -(self.sensors[done_index].position.y ** 2) + (self.anchors[n].position.y ** 2)
                i += 1

            X = np.dot(np.linalg.inv(np.dot(A.transpose(), A)), A.transpose())
            X = np.dot(X, B)
            self.sensors[index].position = Position(X[0][0], X[1][0])



class ImprovedNetworBasickWeighted(Network):
    def set_sensor_distance_to_anchor(self):
        for sensor in self.sensors:

            hop_size = sum([self.anchors[anchor_index].hop_size for anchor_index in range(len(self.anchors))])
            hop_size /= len(self.anchors)

            for i in range(len(self.anchors)):
                if sensor.distance_to_anchor[i] != inf:
                    continue
                assert self.anchors[i].hop_size != inf
                weight = 1 / sensor.hop_to_anchor[i]
                hop_sum = sum([1 / sensor.hop_to_anchor[anchor_index] for anchor_index in range(len(self.anchors))])
                weight /= hop_sum
                sensor.distance_to_anchor[i] = sensor.hop_to_anchor[i] * hop_size


def testNetWork(folder_path: str, dimension, Network, do_plot=False):
    network = Network(folder_path, dimension)
    network.calculate_distance()
    actual_position_path = network.folder_path + "actual_position.csv"
    with open(actual_position_path, 'r') as file:
        data = file.readlines()
        for i in range(len(data)):
            data[i] = data[i].strip()
            data[i] = data[i].split(",")

        assert len(data) == network.dimension
        for i in range(len(data[0])):
            x = float(data[0][i])
            y = float(data[1][i])
            if network.dimension == 3:
                z = float(data[2][i])
                network.sensors[i].actual_position = Position(x, y, z)
            else:
                network.sensors[i].actual_position = Position(x, y)
    if do_plot:
        if dimension == 3:
            x = [sensor.position.x for sensor in network.sensors]
            y = [sensor.position.y for sensor in network.sensors]
            z = [sensor.position.z for sensor in network.sensors]
            ax = plt.gca(projection='3d')
            ax.scatter3D(x, y, z, color='r', marker='+', label="sensor")
            x = [anchor.position.x for anchor in network.anchors]
            y = [anchor.position.y for anchor in network.anchors]
            z = [anchor.position.z for anchor in network.anchors]
            ax.scatter3D(x, y, z, label="anchor", color='b', marker='x')
            x = [sensor.actual_position.x for sensor in network.sensors]
            y = [sensor.actual_position.y for sensor in network.sensors]
            z = [sensor.actual_position.z for sensor in network.sensors]
            ax.scatter3D(x, y, z, color='g', marker='1', label="actual")
            ax.legend()
            # plt.savefig(network.folder_path + "dv_hop.png", dpi=2000)
            plt.show()
        else:
            x = [sensor.position.x for sensor in network.sensors]
            y = [sensor.position.y for sensor in network.sensors]
            plt.scatter(x, y, color='r', marker='+', label="sensor")
            x = [sensor.actual_position.x for sensor in network.sensors]
            y = [sensor.actual_position.y for sensor in network.sensors]
            plt.scatter(x, y, color='g', marker='1', label="actual")
            x = [anchor.position.x for anchor in network.anchors]
            y = [anchor.position.y for anchor in network.anchors]
            plt.scatter(x, y, label="anchor", color='b', marker='x')
            plt.legend()
            plt.savefig("data3_result.png", dpi=600)
            plt.show()

    error_vector = []
    for sensor in network.sensors:
        error_vector.append(sensor.position.distance_to(sensor.actual_position))

    assert len(error_vector) == len(network.sensors)
    return error_vector


class ThreeDimensionNetwork(Network):
    def calculate_distance(self):
        n = len(self.anchors) - 1
        for sensor in self.sensors:
            #  Construct A
            A = np.ndarray([n, 3])
            for i in range(len(A)):
                A[i][0] = self.anchors[i].position.x - self.anchors[n].position.x
                A[i][1] = self.anchors[i].position.y - self.anchors[n].position.y
                A[i][2] = self.anchors[i].position.z - self.anchors[n].position.z
            A = -2 * A
            B = np.ndarray([n, 1])
            for i in range(len(B)):
                assert i != n
                B[i][0] = sensor.distance_to_anchor[i] ** 2 - sensor.distance_to_anchor[n] ** 2
                B[i][0] += -(self.anchors[i].position.x ** 2) + (self.anchors[n].position.x ** 2)
                B[i][0] += -(self.anchors[i].position.y ** 2) + (self.anchors[n].position.y ** 2)
                B[i][0] += -(self.anchors[i].position.z ** 2) + (self.anchors[n].position.z ** 2)
            X = np.dot(np.linalg.inv(np.dot(A.transpose(), A)), A.transpose())
            X = np.dot(X, B)

            sensor.position = Position(X[0][0], X[1][0], X[2])

    def plot(self):
        x = [sensor.position.x for sensor in self.sensors]
        y = [sensor.position.y for sensor in self.sensors]
        z = [sensor.position.z for sensor in self.sensors]
        ax = plt.gca(projection='3d')
        ax.scatter3D(x, y, z, color='r', marker='+', label="sensor")
        x = [anchor.position.x for anchor in self.anchors]
        y = [anchor.position.y for anchor in self.anchors]
        z = [anchor.position.z for anchor in self.anchors]
        ax.scatter3D(x, y, z, label="anchor", color='b', marker='x')
        ax.legend()
        plt.savefig(self.folder_path + "dv_hop.png", dpi=1000)
        plt.show()

    def generate_random_tests(self, sensor_count=None):
        if sensor_count is None:
            sensor_count = len(self.sensors)
        generated_sensors = []
        for i in range(sensor_count):
            x = np.random.uniform(-0.5, 0.5)
            y = np.random.uniform(-0.5, 0.5)
            z = np.random.uniform(-0.5, 0.5)
            sensor = Node(NodeType.Sensor, actual_position=Position(x, y, z))
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

        return TestNetwork3D(test_folder, self.dimension)

class TestNetwork3D(ThreeDimensionNetwork):
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
        ax = plt.gca(projection='3d')
        x = [sensor.position.x for sensor in self.sensors]
        y = [sensor.position.y for sensor in self.sensors]
        z = [sensor.position.z for sensor in self.sensors]
        ax.scatter3D(x, y, z, color='r', marker='+', label="sensor")
        x = [sensor.actual_position.x for sensor in self.sensors]
        y = [sensor.actual_position.y for sensor in self.sensors]
        z = [sensor.actual_position.z for sensor in self.sensors]
        ax.scatter3D(x, y,z, color='g', marker='+', label="actual")

        x = [anchor.position.x for anchor in self.anchors]
        y = [anchor.position.y for anchor in self.anchors]
        z = [anchor.position.z for anchor in self.anchors]
        ax.scatter3D(x, y, z,label="anchor", color='b', marker='x')
        ax.legend()

        plt.show()


class ThreeDimenSionNetWorkImproved(ThreeDimensionNetwork):
    def set_sensor_distance_to_anchor(self):
        for sensor in self.sensors:
            hop_size = 0
            hop_sum = sum([1 / sensor.hop_to_anchor[anchor_index] for anchor_index in range(len(self.anchors))])
            for i in range(len(self.anchors)):
                assert self.anchors[i].hop_size != inf
                weight = 1 / sensor.hop_to_anchor[i]
                weight /= hop_sum
                hop_size += self.anchors[i].hop_size * weight

            for i in range(len(self.anchors)):
                if sensor.distance_to_anchor[i] != inf:
                    continue
                assert self.anchors[i].hop_size != inf
                weight = 1 / sensor.hop_to_anchor[i]
                hop_sum = sum([1 / sensor.hop_to_anchor[anchor_index] for anchor_index in range(len(self.anchors))])
                weight /= hop_sum
                sensor.distance_to_anchor[i] = sensor.hop_to_anchor[i] * hop_size
    def calculate_distance(self):
        done_calculated = set()
        for iter in range(len(self.sensors)):
            #  Construct A
            sensor = self.sensors[iter]
            if sensor.neighbouring_anchor >= 4:
                using_anchors = [self.anchors[temp_i] for temp_i in range(len(self.anchors)) if sensor.hop_to_anchor[temp_i] == 1]
                distances = [sensor.distance_to_anchor[temp_i] for temp_i in range(len(self.anchors)) if sensor.hop_to_anchor[temp_i] == 1]

                using_anchors += [self.sensors[temp_i] for temp_i in done_calculated if temp_i in sensor.distance_to_neighbour.keys() and sensor.distance_to_neighbour[temp_i] != inf and sensor.distance_to_neighbour[temp_i] != 0]
                distances += [sensor.distance_to_neighbour[temp_i] for temp_i in done_calculated if temp_i in sensor.distance_to_neighbour.keys() and sensor.distance_to_neighbour[temp_i] != inf and sensor.distance_to_neighbour[temp_i] != 0]

                assert len(using_anchors) == sensor.neighbouring_anchor
                assert len(distances) == sensor.neighbouring_anchor
                n = len(using_anchors) - 1

                A = np.ndarray([n, 3])

                for i in range(len(A)):
                    A[i][0] = using_anchors[i].position.x - using_anchors[n].position.x
                    A[i][1] = using_anchors[i].position.y - using_anchors[n].position.y
                    A[i][2] = using_anchors[i].position.z - using_anchors[n].position.z
                A = -2 * A
                B = np.ndarray([n, 1])
                for i in range(len(B)):
                    assert i != n
                    B[i][0] = distances[i] ** 2 - distances[n] ** 2
                    B[i][0] += -(using_anchors[i].position.x ** 2) + (using_anchors[n].position.x ** 2)
                    B[i][0] += -(using_anchors[i].position.y ** 2) + (using_anchors[n].position.y ** 2)
                    B[i][0] += -(using_anchors[i].position.z ** 2) + (using_anchors[n].position.z ** 2)
                try:
                    X = np.dot(np.linalg.inv(np.dot(A.transpose(), A)), A.transpose())
                except np.linalg.LinAlgError as e:
                    print(e)
                X = np.dot(X, B)

                sensor.position = Position(X[0][0], X[1][0], X[2][0])
                done_calculated.add(iter)
                for index in range(len(self.sensors)):
                    if index == iter:
                        continue
                    if index in sensor.distance_to_neighbour.keys() and self.sensors[index].distance_to_neighbour[iter] != inf and self.sensors[index].distance_to_neighbour[iter] != 0:
                        self.sensors[index].neighbouring_anchor += 1
        n = len(self.anchors) - 1
        for index in range(len(self.sensors)):
            if index in done_calculated:
                continue

            additional_using_anchors = set()
            for calculated_index in done_calculated:
                if index in self.sensors[calculated_index].distance_to_neighbour.keys():
                    if self.sensors[calculated_index].distance_to_neighbour[index] != inf and \
                            self.sensors[calculated_index].distance_to_neighbour[index] != 0:
                        additional_using_anchors.add(calculated_index)
            additional_n = len(additional_using_anchors)
            A = np.ndarray([n + additional_n, 3])
            i = 0

            while i < n:
                A[i][0] = self.anchors[i].position.x - self.anchors[n].position.x
                A[i][1] = self.anchors[i].position.y - self.anchors[n].position.y
                A[i][2] = self.anchors[i].position.z - self.anchors[n].position.z
                i += 1
            for done_index in additional_using_anchors:
                A[i][0] = self.sensors[done_index].position.x - self.anchors[n].position.x
                A[i][1] = self.sensors[done_index].position.y - self.anchors[n].position.y
                A[i][2] = self.sensors[done_index].position.z - self.anchors[n].position.z
                i += 1
            assert i == n + additional_n
            A = -2 * A
            B = np.ndarray([n + additional_n, 1])
            i = 0
            while i < n:
                B[i][0] = self.sensors[index].distance_to_anchor[i] ** 2 - self.sensors[index].distance_to_anchor[
                    n] ** 2
                B[i][0] += -(self.anchors[i].position.x ** 2) + (self.anchors[n].position.x ** 2)
                B[i][0] += -(self.anchors[i].position.y ** 2) + (self.anchors[n].position.y ** 2)

                B[i][0] += -(self.anchors[i].position.z ** 2) + (self.anchors[n].position.z ** 2)
                i += 1

            for done_index in additional_using_anchors:
                B[i][0] = self.sensors[index].distance_to_neighbour[done_index] ** 2 - \
                          self.sensors[i].distance_to_anchor[
                              n] ** 2
                B[i][0] += -(self.sensors[done_index].position.x ** 2) + (self.anchors[n].position.x ** 2)
                B[i][0] += -(self.sensors[done_index].position.y ** 2) + (self.anchors[n].position.y ** 2)
                B[i][0] += -(self.sensors[done_index].position.z ** 2) + (self.anchors[n].position.z ** 2)
                i += 1

            X = np.dot(np.linalg.inv(np.dot(A.transpose(), A)), A.transpose())
            X = np.dot(X, B)
            self.sensors[index].position = Position(X[0][0], X[1][0], X[2][0])


if __name__ == '__main__':
    # network = Network("./data-csv/data1/", 2)
    # network.calculate_distance()
    # network.plot()
    # neighbouring = [sensor.neighbouring_anchor for sensor in network.sensors]
    # print(neighbouring)
    # test_network = network.generate_random_tests()
    #
    # errors = test_network.calculate_error()
    # test_network.plot()
    # average_error = np.mean(errors)


    # #  Test 2D networks
    # test_path = "./data-csv/data1/test3/"
    # errors = testNetWork(test_path, 2, Network)
    # average_error = np.mean(errors)
    # print(average_error)
    #
    # errors = testNetWork(test_path, 2, ImprovedNetworkWeighted)
    # average_error = np.mean(errors)
    # print(average_error)
    #
    # errors = testNetWork(test_path, 2, NewImprovedNetworkCalculateSomePointsFirst)
    # average_error = np.mean(errors)
    # print(average_error)


    #  Test 3D network
    test_path = "./data-csv/data4/test2/"
    errors = testNetWork(test_path, 3, ThreeDimensionNetwork)
    average_error = np.mean(errors)
    print(average_error)

    errors = testNetWork(test_path, 3, ThreeDimenSionNetWorkImproved)
    average_error = np.mean(errors)
    print(average_error)
