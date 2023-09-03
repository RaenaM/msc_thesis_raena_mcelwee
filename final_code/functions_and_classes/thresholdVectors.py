import numpy as np

class ThresholdVectors():

    def __init__(self, 
                 matrix):
        self.matrix = matrix

    def standard_deviation(self,
                           c: float = 0.5):
        # assert c > 0, 'c must be a postive number'
        threshold = c * np.std(self.matrix, axis = 0, ddof = 1)
        return np.array(threshold)