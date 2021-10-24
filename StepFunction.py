import numpy as np


class StepFunction:

    def get_value(self, value):
        if value > 0:
            return 1
        else:
            return 0


    def get_derivative(self, value):
        return np.array([1])


