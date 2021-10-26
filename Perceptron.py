import sys

import numpy as np
from sklearn.utils import shuffle


class Perceptron:

    BIAS = 2

    def __init__(self, features_amount, activation_function):
        self.weights = np.random.rand(1, features_amount + 1)  # cantidad de entradas (x1,..,xn) mas un lugar para la constante
        self.min_weights = self.weights
        self.error = sys.maxsize
        self.min_error = sys.maxsize
        self.activation_function = activation_function
        self.delta = np.zeros(features_amount + 1)
        self.restart_condition = 0
        self.restart_count = 0
        self.learning_rate = 0
        self.last_activation_value = 0

    def set_weights(self, weights):
        w = [weights]
        self.weights = np.array(w)

    def run(self, training_example, run_mode):
        x = np.array(training_example[:-1])  # training menos label
        x = np.append(x, [1])  # agrego el 1 del bias
        excitement = np.dot(self.weights, x.transpose())
        activation = self.activation_function.get_value(excitement[0])

        error = classification_error(activation, training_example[-1])
        if run_mode == "training":
            # * self.activation_function.get_derivative(excitement[0])
            delta = self.learning_rate * (training_example[-1] - activation) * x
            return delta, error
        else:
            return activation, error

    def check_restart(self, restart_condition):
        if self.restart_count > restart_condition:
            self.weights = np.random.rand(1, len(self.weights[0]))

    # TODO checkar lo de la recta con el cuaderno
    def get_line(self):
        weights = self.weights[0]
        return {'m': -weights[0] / weights[1], 'b': -weights[self.BIAS] / weights[1]}

    def batch_training(self, training_set, learning_rate, restart_condition, iteration_limit, need_to_normalize=False):  # matriz el conjunto)
        self.restart_condition = restart_condition
        self.learning_rate = learning_rate
        iteration_count = 0
        errors_per_epoch = []
        while self.error > 0 and iteration_count < iteration_limit:
            self.check_restart(restart_condition)
            self.error = 0
            self.delta = np.zeros(len(training_set[0]))
            for training_example in training_set:
                delta, error = self.run(training_example, "training")
                self.error += error
                # self.delta += delta # esto no funciona por cli. solo dios sabe y un chino en @see https://blog.csdn.net/weixin_39278265/article/details/85148974
                np.add(self.delta, delta, out=self.delta, casting="unsafe")
            self.weights = np.add(self.weights, self.delta)
            self.min_error, self.min_weights, self.restart_count, errors_per_epoch = error_handling(errors_per_epoch, self.error, iteration_count, self.min_error, self.min_weights, self.weights, self.restart_count)
            iteration_count += 1
        return self.min_weights, errors_per_epoch

    def incremental_training(self, training_set, learning_rate, restart_condition, iteration_limit, need_to_normalize=False):
        self.restart_condition = restart_condition
        self.learning_rate = learning_rate
        iteration_count = 0
        errors_per_epoch = []
        while self.error > 0 and iteration_count < iteration_limit:
            self.check_restart(restart_condition)
            self.error = 0
            training_set = shuffle(training_set)
            for training_example in training_set:
                self.delta, error = self.run(training_example, "training")
                self.error += error
                self.weights = np.add(self.weights, self.delta)
            self.min_error, self.min_weights, self.restart_count, errors_per_epoch = error_handling(errors_per_epoch, self.error, iteration_count, self.min_error, self.min_weights, self.weights, self.restart_count)
            iteration_count += 1
        return self.min_weights, errors_per_epoch

    def test_perceptron(self, testing_set, silent=False):
        halfwaySquareError = 0
        self.error = 0
        for testing_example in testing_set:
            output, error = self.run(testing_example, "testing")
            if not silent:
                print("neuron answer for parameters: ", np.array_str(testing_example), " is ", output, " real answer is ", testing_example[-1])
            self.error += error
            halfwaySquareError += np.square(testing_example[-1] - output)
        print("n=", len(testing_set))
        return self.error, halfwaySquareError/2


def error_handling(errors, error, iteration, min_error, min_weights, weights, cost_count):
    errors.append(error)  # cuantos se clasificaron mal en esta epoca
    if error < min_error:
        min_error = error
        min_weights = weights
    if iteration != 0:
        if errors[iteration] == errors[iteration - 1]:
            cost_count += 1
        else:
            cost_count = 0
    return min_error, min_weights, cost_count, errors


def classification_error(real_output, desired_output):
    if real_output - desired_output:  # lo clasifico mal
        return 1
    else:
        return 0