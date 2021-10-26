import math


def distance_to_plane(x1, y1, w0, w1, bias):
    d = abs((w0 * x1 + w1 * y1 + bias)) / (math.sqrt(w0 * w0 + w1 * w1))
    return d


def line_for_points(x1, y1, x2, y2):
    line = {'m': (y1 - y2) / (x1 - x2), 'b': (x1 * y2 - x2 * y1) / (x1 - x2)}
    return line


def line_for_weights(w0, w1, w2):
    return {'m': -w0 / w1, 'b': -w2 / w1}


def line_for_points_w(class_2_point, class_2_point_b, class_1_point):
    w1 = - (class_2_point[1] - class_2_point_b[1])
    w2 = class_2_point[0] - class_2_point[0]

    w0 = -(class_2_point[0] * w1 + class_2_point[1] * w2 + class_1_point[0] * w1 + class_1_point[1] * w2)/2
    return {'w0': w0, 'w1': w1, 'w2': w2}


def get_optimal_planes_candidates(class_1_points, class_2_points):
    candidates = []
    for class_2_point in class_2_points:
        for class_2_point_b in class_2_points:
            if class_2_point[0] != class_2_point_b[0] and class_2_point[1] != class_2_point_b[1]:
                for class_1_point in class_1_points:
                    line = line_for_points_w(class_2_point, class_2_point_b, class_2_point_b)
                    d = distance_to_plane(class_1_point[0], class_1_point[1], line['w0'], line['w1'], line['w2'])
                    line['d'] = d
                    candidates.append(line)
    return candidates
