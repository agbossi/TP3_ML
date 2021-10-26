import random

import numpy as np
import matplotlib.pyplot as plt
from utils import Point


def rest_points(point_a, point_b):
    return Point(point_a.x - point_a.x, point_b.y - point_b.y, point_a.type)


def get_x2(w, x1):
    return (-w[0]-x1*w[1])/w[2]


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

def generate_plane(a1, a2, b1):
    nx = - (a1.y - a2.y)
    ny = a1.x - a2.x

    w0 = -(a1.x * nx + a1.y * ny + b1.x * nx + b1.y * ny)/2

    plane = Plane([w0, nx, ny])
    dist = plane.distance_to_plane(b1)

    return plane, abs(dist)

point_1 = Point(random.normalvariate(0, 2), random.normalvariate(0, 2), 1)
point_2 = Point(random.normalvariate(0, 2), random.normalvariate(0, 2), 1)
point_3 = Point(random.normalvariate(0, 2), random.normalvariate(0, 2), 2)

plane, dist = generate_plane(point_1, point_2, point_3)

plt.plot(point_1.x, point_1.y, 'ro')
plt.plot(point_2.x, point_2.y, 'ro')
plt.plot(point_3.x, point_3.y, 'bo')

#plt.plot([0, nx], [0, ny], '-')

points_x, points_y = plane.compute_points()

print("dist = %0.2f" % dist)
plt.plot(points_x, points_y)
plt.show()

