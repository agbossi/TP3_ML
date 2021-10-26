import pickle
import random
from utils import Point
import matplotlib.pyplot as plt


def generate_points(mu, sigma, point_type, num_points):
    points = []

    for i in range(num_points):
        x_pos = random.normalvariate(mu[0], sigma[0])
        y_pos = random.normalvariate(mu[1], sigma[1])

        points.append(Point(x_pos, y_pos, point_type))

    return points


def save_points(points, filename):
    pickle_string = pickle.dumps(points)
    file = open(filename, "wb")
    file.write(pickle_string)


def load_points(filename):
    file = open(filename, "rb")
    pickle_string = file.read()
    return pickle.loads(pickle_string)


def generate_points():
    points = []

    points_min = [0, 0]
    points_max = [5, 5]


    points_0 = generate_points(
        mu = [3.5,3.5],
        sigma = [0.2, 0.2],
        point_type = 1,
        num_points = 50)

    points_1 = generate_points(
        mu = [2.5,2.5],
        sigma = [0.2, 0.2],
        point_type = -1,
        num_points = 50
    )

    total_points = points_0 + points_1

    for point in total_points:
       # print(point.x, point.y, point.type)
        plt.plot(point.x, point.y, ('ro' if point.type == 1 else 'bo'))


    plt.title("Conjunto TP3-1")

    plt.show()

    save_points(total_points, filename="points/TP3-1.txt")

    ## Points

    points_0 = generate_points(
        mu = [3.5,3.5],
        sigma = [0.3, 0.3],
        point_type = 1,
        num_points = 50)

    points_1 = generate_points(
        mu = [2.5,2.5],
        sigma = [0.43, 0.3],
        point_type = -1,
        num_points = 50
    )

    total_points = points_0 + points_1

    for point in total_points:
       # print(point.x, point.y, point.type)
        plt.plot(point.x, point.y, ('ro' if point.type == 1 else 'bo'))


    plt.title("Conjunto TP3-2")

    plt.show()

    save_points(total_points, filename="points/TP3-2.txt")

if __name__ == "__main__":
    generate_points()