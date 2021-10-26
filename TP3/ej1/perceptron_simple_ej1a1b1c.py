import matplotlib.pyplot as plt
import numpy as np
from random import randrange

import generate_points
from utils import Plane


counter = 0

def signo(x):
    return 1 if x > 0 else -1


def calcular_error(points, w):
    error = 0
    for point in points:
        h = w[0]
        for i in range(len(point.coords)):
            h += w[i+1] * point.coords[i]

        y_estimado = signo(h)
        error += point.type!=y_estimado

    return error / len(points)


def get_nearest_points(points, plane, nearest_cant=5):
    red_points = []
    blue_points = []
    #print(points)

    for point in points:
        #print(point)
        if point.type == 1:
            red_points.append([point, plane.distance_to_plane(point)])
        else:
            blue_points.append([point, plane.distance_to_plane(point)])

    red_points.sort(reverse=False, key=lambda e: e[1])
    blue_points.sort(reverse=True, key=lambda e: e[1])

    return red_points[:nearest_cant], blue_points[:nearest_cant]


def plot_and_save(basic_plane, plane, total_points, support_vectors):
    for point in total_points:
        # print(point.x, point.y, point.type)
        plt.plot(point.x, point.y, ('ro' if point.type == 1 else 'bo'))

    for point in support_vectors:
        plt.plot(point.x, point.y, 'yo')


    x_points, y_points = basic_plane.compute_points()
    plt.plot(x_points, y_points, 'b-', label="hiperplano perceptron simple")

    x_points, y_points = plane.compute_points()
    plt.plot(x_points, y_points, 'r-', label="hiperplano optimo")

    global counter
    print("saving planes/%d.png" % counter)

    plt.legend()
    plt.savefig("planes/%d.png" % counter, dpi=1200)
    plt.clf()

    counter += 1


def compute_plane_pairs(red_points, blue_points, total_points, basic_plane):
    min_distance = 1e9
    selected_plane = None

    # # take 1 red and 2 blue
    for point_a in red_points:
        for point_b in red_points:
            for point_c in blue_points:
                pass

    plane, dist = generate_plane(red_points[0][0], red_points[1][0], blue_points[0][0])
    if dist < min_distance:
        min_distance = dist
        selected_plane = plane

    print(total_points)

    sel_a = None
    sel_b = None
    sel_c = None

    # take 2 red and 1 blue
    for point_a in red_points:
        for point_b in blue_points:
            for point_c in blue_points:
                if point_b[0] != point_c[0]:
                    plane, dist = generate_plane(point_b[0], point_c[0], point_a[0])

                    #plot_and_save(basic_plane, plane, total_points, [point_a[0], point_b[0], point_c[0]])

                    print("error del plano = %f, dist = %f " % (calcular_error(total_points, plane.w), dist))
                    if calcular_error(total_points, plane.w) < 1e-6:
                        if dist < min_distance:
                            print("plane selected")
                            min_distance = dist
                            selected_plane = plane
                            sel_a = point_a[0]
                            sel_b = point_b[0]
                            sel_c = point_c[0]

    return selected_plane, min_distance, sel_a, sel_b, sel_c


def perceptron_simple(points, iteraciones, learning_rate):
    it = 0
    w = np.zeros(len(points[0].coords)+1)

    error_min = len(points[0].coords)*2
    w_min = w

    while error_min > 0 and it < iteraciones:
        it += 1

        x_point = points[randrange(len(points))].coords
        y_point = points[randrange(len(points))].type

        h = w[0]

        for i in range(len(x_point)):
            h += w[i+1] * x_point[i]

        y_clasificado = signo(h)

        delta_w = []

        delta_w.append(learning_rate * (y_point - y_clasificado) )

        for x in range(len(x_point)):
            delta_w.append(learning_rate * (y_point - y_clasificado) * x_point[x])

        for x in range(len(delta_w)):
            w[x] += delta_w[x]

        error_w = calcular_error(points, w)

        #print(w, "=>", error_w)

        if error_w < error_min:
            error_min = error_w
            w_min = w
    print("Conseguimos el hiperplano de separacion en %d iteraciones con error %f" % (it, error_min))

    return Plane(w_min)


total_points = generate_points.load_points("points/TP3-1.txt")

for point in total_points:
    # print(point.x, point.y, point.type)
    plt.plot(point.x, point.y, ('ro' if point.type == 1 else 'bo'))



### Mostramos el plano inicial

plane_simple = perceptron_simple(total_points, iteraciones=10000, learning_rate=0.1)

x_points, y_points = plane_simple.compute_points()

#plt.plot(x_points, y_points, "b-")



red_points, blue_points = get_nearest_points(total_points, plane_simple)

for point in red_points:
    plt.plot(point[0].x, point[0].y, 'yx')

for point in blue_points:
    plt.plot(point[0].x, point[0].y, 'yx')

plane, distance, sel_a, sel_b, sel_c = compute_plane_pairs(red_points, blue_points, total_points, plane_simple)


print(distance)

plt.plot(sel_a.x, sel_a.y, 'gx')
plt.plot(sel_b.x, sel_b.y, 'gx')
plt.plot(sel_c.x, sel_c.y, 'gx')

x_points, y_points = plane_simple.compute_points()
plt.plot(x_points, y_points, 'b-', label="hiperplano perceptron simple")

x_points, y_points = plane.compute_points()
plt.plot(x_points, y_points, 'r-', label="hiperplano optimo")

plt.legend()
plt.show()



total_points = generate_points.load_points("points/TP3-2.txt")

for point in total_points:
    # print(point.x, point.y, point.type)
    plt.plot(point.x, point.y, ('ro' if point.type == 1 else 'bo'))

### version con TP3-2

plane_simple = perceptron_simple(total_points, iteraciones=10000, learning_rate=0.1)
x_points, y_points = plane.compute_points()
plt.plot(x_points, y_points, 'r-', label="hiperplano perceptron simple")
plt.legend()
plt.show()



