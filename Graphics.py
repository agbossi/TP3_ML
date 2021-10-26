import matplotlib.pyplot as plt
import numpy as np


def converge_metric(iterations, errors):
    epochs = np.arange(1, iterations+1)
    errors = np.append(errors, np.zeros(len(epochs) - len(errors)))
    plt.plot(epochs, errors)
    plt.xlabel('iterations')
    plt.ylabel('errors')
    plt.show()


def plot_line(x, y, line, show=True, linestyle='dotted'):
    plt.scatter(x[:, 0], x[:, 1], c=y)
    plt.plot(x[:, 0], x[:, 0] * line['m'] + line['b'], linestyle=linestyle)
    if show:
        plt.show()
    return plt.xlim(), plt.ylim()


def plot_closest_points(class_1_points, class_2_points, x, y, line, show=True):
    xlim, ylim = plot_line(x, y, line, show=False)
    plt.scatter(class_1_points[:, 0], class_1_points[:, 1], marker='d')
    plt.scatter(class_2_points[:, 0], class_2_points[:, 1], marker='v')
    plt.xlabel("x1")
    plt.ylabel("x2")


    plt.xlim(xlim)
    plt.ylim(ylim)
    if show:
        plt.title("Puntos mas cercanos a la recta del perceptron")
        plt.show()
    return xlim, ylim


def plot_candidates(candidates, x, y):
    for candidate in candidates:
        plot_line(x, y, candidate, show=False,linestyle='solid')
    plt.show()


def plot_separators(class_1_points, class_2_points, x, y, line, optimal_separator):
    xlim, ylim = plot_closest_points(class_1_points, class_2_points, x, y, line, show=False)
    plt.plot(x[:, 0], x[:, 0] * optimal_separator['m'] + optimal_separator['b'], linestyle='solid')
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.title("separacion optima vs perceptron")
    plt.show()
