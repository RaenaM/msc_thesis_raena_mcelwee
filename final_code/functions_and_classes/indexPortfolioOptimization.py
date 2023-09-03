import scipy
import numpy as np
from costFunctions import CostFunctions

class IndexPortfolioOptimisation(object):

    def __init__(self,
                 cost_function: CostFunctions,
                 min_weight: float = 0.0,
                 max_weight: float = 1.0, # consider writing a version that takes a vector
                 factor: float = 252.0):
        """Get a vector of weights with portfolio optimization.

        Args:
            cost_function (CostFunctions): _description_
            min_weight (float, optional): Minimum value any single weight can take on, along with the constraint that all weights sum to one. Defaults to 0.0.
            max_weight (float, optional): Maximum value any single weight can take on, along with the constraint that all weights sum to one. Defaults to 1.0.
            factor (float, optional): Factor to multiply by (daily: 252.0; monthly: 12.0) Defaults to 252.0.
        """
        self.cost_function = cost_function
        self.min_weight = min_weight # maybe consider settings restrictions on max that can be invested into different asset classes...
        self.max_weight = max_weight
        self.constraints = [{'type' : 'eq', 
                             'fun' : lambda w: np.sum(w) - 1.0}]
        self.factor = factor

    def mean_variance(self,
                      covMatrix,
                      expReturns,
                      targetStandDev,
                      current_assets,
                      prev_weights = None):
        
        assert type(covMatrix) == np.ndarray, "`covMatrix` needs to be of type numpy.ndarray"
        assert type(expReturns) == np.ndarray, "`expReturns` needs to be of type numpy.ndarray"

        self.K = covMatrix.shape[0]

        self.prev_weights = prev_weights
        if self.prev_weights is None:
            self.prev_weights = {asset: 0 for asset in current_assets}

        costFunc = self.cost_function.execution(prev_weights = self.prev_weights, current_assets = current_assets)
        self.covMatrix = covMatrix.copy() * self.factor 
        self.expReturns = expReturns.copy() * self.factor
        self.targetStandDev = targetStandDev

        def optim_function(w, mu = self.expReturns, costFunc = costFunc): 
            return costFunc(w) - float(np.sum(mu * w))
        self.optim_function = optim_function

        def variance_constraint(w, 
                                target = self.targetStandDev, 
                                cov = self.covMatrix): 
             return target - self.annualised_portfolio_standard_deviation(cov, w = w)

        self.constraints.append({'type' : 'ineq',
                                 'fun' : lambda w: variance_constraint(w)})
        return self
    
    def minimum_variance(self,
                         covMatrix,
                         prev_weights = None):
        
        assert type(covMatrix) == np.ndarray, "`covMatrix` needs to be of type numpy.ndarray"

        self.prev_weights = prev_weights

        self.K = covMatrix.shape[0]
        self.covMatrix = covMatrix.copy() * self.factor

        def optim_function(w, covMatrix = self.covMatrix): return np.sqrt(np.dot(w.T, np.dot(covMatrix, w)))
        self.optim_function = optim_function

        return self

    def fit(self,
            init_weights = None):
        
        init_weights = np.ones(self.K)/self.K
        bounds = tuple((self.min_weight, self.max_weight) for k in range(self.K))

        try:
            weights = scipy.optimize.minimize(self.optim_function, 
                                              x0 = init_weights, 
                                              constraints = self.constraints, 
                                              bounds = bounds,
                                              method = "SLSQP")
        except: 
            weights = scipy.optimize.minimize(self.optim_function, 
                                              x0 = init_weights,
                                              constraints = self.constraints, 
                                              method = "trust-constr")
        return np.array(weights.x)

    def annualised_portfolio_standard_deviation(self, cov, w): 
        portVariance = np.dot(w.T, np.dot(cov, w))
        if portVariance <= 0:
            portVariance = 1e-20
        portStandardDev = float(np.sqrt(portVariance))
        return portStandardDev