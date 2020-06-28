import pandas as pd
from Network import *


def draw_image(error1, error2):
    size = 4
    x = ['data1', 'data2', 'data3', 'data4']
    a = np.random.random(size)
    b = np.random.uniform(0, a, size=size)

    plt.bar(x, error1, label='Average error for model 1')
    plt.bar(x, error2, label='Average error for model 2')
    plt.legend()
    plt.savefig("error_image.png", dpi=1000)
    plt.show()


if __name__ == '__main__':
    filename = "./data-csv/data2/"
    dimension = 2
    network = Network(filename, dimension)
    # network.calculate_distance()
    # network.plot()
    x = []
    y = []
    for i in range(20,100,5):
        temp_error = []
        for _ in range(10):
            while True:
                try:
                    test = network.generate_random_tests(i)
                except AssertionError as e:
                    continue
                break
            errors = testNetWork(test.folder_path, 2, NewImprovedNetworkCalculateSomePointsFirst)
            temp_error.append(np.mean(errors))
        x.append(i)
        y.append(np.mean(temp_error))

    plt.plot(x,y)
    plt.xlabel("number of sensors")
    plt.ylabel("Average error")
    plt.savefig("iteration.png", dpi=1000)
    plt.show()
    # folder = "./data-csv/data2/test6"
    # dimension = 2
    # errors = testNetWork(folder, dimension, NewImprovedNetworkCalculateSomePointsFirst)
    # 1-5 20
    # 6-10 30
    # 11-15 40
    # test = network.generate_random_tests()
    #
    # errors = testNetWork(test.folder_path, test.dimension, ImprovedNetworkWeighted)
    # print(np.mean(errors))
    #
    # errors = testNetWork(test.folder_path, test.dimension, Network)
    # print(np.mean(errors))

    print(np.mean(errors))
    # error1 = []
    # error2 = []
    # for j in range(1,5):
    #     total_error = []
    #     for i in range(1,6):
    #         filename = "./data-csv/data{}/test{}/".format(j,i)
    #         if j == 4:
    #             errors = testNetWork(filename, 3, ThreeDimensionNetwork)
    #         else:
    #             errors = testNetWork(filename, 2, Network)
    #
    #         total_error.append(np.mean(errors))
    #     error1.append(np.mean(total_error))
    #
    # for j in range(1,5):
    #     total_error = []
    #     for i in range(1,6):
    #         filename = "./data-csv/data{}/test{}/".format(j,i)
    #         if j == 4:
    #             errors = testNetWork(filename, 3, ThreeDimenSionNetWorkImproved)
    #         else:
    #             errors = testNetWork(filename, 2, NewImprovedNetworkCalculateSomePointsFirst)
    #         total_error.append(np.mean(errors))
    #     error2.append(np.mean(total_error))
    #
    # draw_image(error1, error2)
