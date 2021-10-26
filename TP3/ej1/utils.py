import numpy as np


def get_x2(w, x1):
    return (-w[0]-x1*w[1])/w[2]


class Point:
    def __init__(self, x, y, type):
        self.x = x
        self.y = y
        self.coords = x, y
        self.type = type


class Plane:
    def __init__(self, w):
        self.w = w

    def distance_to_plane(self, point):
        return self.w[0] + self.w[1] * point.x + self.w[2] * point.y

    def compute_points(self):
        x_points_pre = np.linspace(-5, 5, 100)

        y_points = []
        x_points = []

        for x1 in x_points_pre:
            y = get_x2(self.w, x1)

            if y > -5 and y < 5:
                x_points.append(x1)
                y_points.append(y)

        return x_points_pre, y_points


def get_X_from_points(points):
    X = []
    for point in points:
        X.append([point.x, point.y])
    return X


def get_Y_from_points(points):
    y = []
    for point in points:
        y.append([point.type])
    return y