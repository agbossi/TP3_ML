import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, svm
from Perceptron import Perceptron
from StepFunction import StepFunction
import geometricHelper as geom
import Graphics as mtr
from mlxtend.plotting import plot_decision_regions


def svm_classification(x, y):
    SVM = svm.SVC(kernel='linear', degree=3, C=1000)
    SVM.fit(x, y)
    plot_decision_regions(x, y, clf=SVM)
    plt.title("SVM sobre conjunto")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()


def fill_points(points, class_distances, support_points):
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
        d = geom.distance_to_plane(point[0], point[1], w0, w1, bias)
        if point[2] == 1:
            class_1_distances[i] = d
        else:
            class_2_distances[i] = d
    class_1_points = fill_points(points, class_1_distances, support_points)
    class_2_points = fill_points(points, class_2_distances, support_points)
    return class_1_points, class_2_points


def get_optimal_plane(candidates, dataset, perceptron, nearest_points):
    optimal_separator = None
    min_max_distance = 0

    for candidate in candidates:
        perceptron.set_weights([candidate['w0'], candidate['w1'], candidate['w2']])
        error, halfwaySquareError = perceptron.test_perceptron(dataset)
        if error == 0:
            distances = []
            for point in nearest_points:
                weights = perceptron.weights[0]
                distances.append(geom.distance_to_plane(point[0], point[1], weights[0], weights[1], weights[2]))
            distances.sort()
            if min_max_distance < distances[0]:
                min_max_distance = distances[0]
                optimal_separator = candidate
    return optimal_separator


# lineal
x_l, y_l = datasets.make_blobs(n_samples=4, n_features=2, centers=[(1, 1), (3, 3)], cluster_std=0.6)
# no lineal
x_nl, y_nl = datasets.make_blobs(n_samples=20, n_features=2, centers=[(1, 1), (2, 2)], cluster_std=1.2)
x = x_l
y = y_l

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

candidates = geom.get_optimal_planes_candidates(class_1_points, class_2_points)
for candidate in candidates:
    mtr.plot_closest_points(class_1_points, class_2_points, x, y, geom.line_for_weights(candidate['w0'], candidate['w1'], candidate['w2']))
closest_points = np.vstack([class_1_points, class_2_points])
optimal_separator = get_optimal_plane(candidates, dataset, p, closest_points)
mtr.plot_separators(class_1_points, class_2_points, x, y, line, optimal_separator)

svm_classification(x, y)
