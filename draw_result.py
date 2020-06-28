from Network import *

if __name__ == '__main__':
    folder = "./data-csv/data4/"
    dimension = 3
    network = ThreeDimenSionNetWorkImproved(folder, dimension)
    network.calculate_distance()
    network.plot()