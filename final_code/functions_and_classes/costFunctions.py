import numpy as np

class CostFunctions(object):

    def __init__(self,
                 transaction_cost: float = 0.001):
        self.transaction_cost = transaction_cost

    def execution(self, prev_weights, current_assets):
        self.prev_weights = prev_weights
        N = len(current_assets)
        pw = np.zeros(N)
        for i in range(N):
            asset = current_assets[i]
            if asset in prev_weights.keys():
                pw[i] = prev_weights[asset]
        self.pw = pw
        def execution_cost_function(w, PW = self.pw):
            return self.transaction_cost * np.sum(abs(w - PW))
        return execution_cost_function