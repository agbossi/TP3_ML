import matplotlib.pyplot as plt
import numpy as np


def converge_metric(iterations, errors):
    epochs = np.arange(1, iterations+1)
    errors = np.append(errors, np.zeros(len(epochs) - len(errors)))
    plt.plot(epochs, errors)
    plt.xlabel('iterations')
    plt.ylabel('errors')
    plt.show()


def plot_line(x, y, line, show=True):
    plt.scatter(x[:, 0], x[:, 1], c=y)
    plt.plot(x[:, 0], x[:, 0] * line['m'] + line['b'], linestyle='dashed')
    if show:
        plt.show()
    return plt.xlim(), plt.ylim()


def plot_closest_points(class_1_points, class_2_points, x, y, line):
    xlim, ylim = plot_line(x, y, line, show=False)
    plt.scatter(class_1_points[:, 0], class_1_points[:, 1], marker='d')
    plt.scatter(class_2_points[:, 0], class_2_points[:, 1], marker='v')
    plt.title("Puntos mas cercanos a la recta del perceptron")
    plt.xlabel("x1")
    plt.ylabel("x2")


    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.show()