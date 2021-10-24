import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from Perceptron import Perceptron
from StepFunction import StepFunction
import Graphics as mtr


def distance_to_plane(x1, y1, w0, w1, bias):
    d = abs((w0 * x1 + w1 * y1 + bias)) / (math.sqrt(w0 * w0 + w1 * w1))
    return d


def tandas(puntos_min_1, puntos_min_2):
    tanda_1 = []
    tanda_1.append([puntos_min_1[0], puntos_min_1[1]])
    tanda_1.append([puntos_min_1[0], puntos_min_1[2]])
    tanda_1.append([puntos_min_1[1], puntos_min_1[2]])
    tanda_1 = np.array(tanda_1)

    tanda_2 = []
    tanda_2.append([puntos_min_2[0], puntos_min_2[1]])
    tanda_2.append([puntos_min_2[0], puntos_min_2[2]])
    tanda_2.append([puntos_min_2[1], puntos_min_2[2]])
    tanda_2 = np.array(tanda_2)

    return tanda_1, tanda_2


def fill_points(points, class_distances):
    class_distances = {k: v for k, v in sorted(class_distances.items(), key=lambda item: item[1])}
    class_indexes = list(class_distances.keys())[0:support_points]
    class_points = []
    for index in class_indexes:
        class_points.append(points[index])
    return np.array(class_points)


def get_support_points(points, w0, w1, bias, support_points):
    class_1_distances = {}
    class_2_distances = {}
    for i in range(len(points)):
        point = points[i]
        d = distance_to_plane(point[0], point[1], w0, w1, bias)
        if point[2] == 1:
            class_1_distances[i] = d
        else:
            class_2_distances[i] = d
    class_1_points = fill_points(points, class_1_distances)
    class_2_points = fill_points(points, class_2_distances)
    return class_1_points, class_2_points


x, y = datasets.make_blobs(n_samples=15, n_features=2, centers=[(1, 1), (3, 3)], cluster_std=0.6)
plt.scatter(x[:, 0], x[:, 1], c=y)
plt.show()

activation_function = StepFunction()
features = 2
iteration_limit = 100
restart_condition = 100
learning_rate = 0.1

p = Perceptron(features, activation_function)
dataset = np.column_stack((x[:, 0], x[:, 1], y))

trained_weights, errors_per_epoch = p.incremental_training(dataset, learning_rate, restart_condition, iteration_limit)
mtr.converge_metric(iteration_limit, errors_per_epoch)

line = p.get_line()
mtr.plot_line(x, y, line)

support_points = 3
class_1_points, class_2_points = get_support_points(dataset, p.weights[0][0], p.weights[0][1], p.weights[0][2], support_points)
mtr.plot_closest_points(class_1_points, class_2_points, x, y, line)


# p.test_perceptron(and_data_set)
