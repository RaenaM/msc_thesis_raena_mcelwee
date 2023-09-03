import numpy as np

class ExpectedReturns():

    def __init__(self, 
                 returnsMatrix):
        self.returnsMatrix = np.array(returnsMatrix)

    def meanHistoricalReturns(self):
        """Mean Historical Returns

        Returns:
            nmupy.ndarray: returns a vector of sample mean returns
        """
        return np.mean(self.returnsMatrix, axis = 0)
