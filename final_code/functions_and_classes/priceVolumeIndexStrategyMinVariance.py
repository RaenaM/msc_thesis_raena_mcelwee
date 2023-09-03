from indexFeatureEngineering import *
from covarianceMatrixVolume import *
from costFunctions import *
from indexPortfolioOptimization import *
from thresholdVectors import *
from expectedReturns import *
from tqdm.auto import tqdm
import pandas as pd

class PriceVolumeIndexMinVarStrategy(IndexFeatureEngineering,
                                    ThresholdVectors,
                                    CovarianceMatrixVolume,
                                    CostFunctions,
                                    IndexPortfolioOptimisation):
    
    def __init__(self,
                 indexDict: dict,
                 trading_days: pd.core.indexes.datetimes.DatetimeIndex,
                 priceDict: dict,
                 volumeDict: dict,
                 numberOfMembers: int = 10,
                 indexWeightCol: str = 'Weight',
                 indexMemberCol: str = 'INDX_MWEIGHT_HIST',
                 returnsColumn: str = 'PX_LAST',
                 volumeColumn: str = 'PX_VOLUME',
                 getLogReturns: bool = True,
                 logVolume: bool = True,
                 standardizeVolume: bool = True,
                 lookbackWindow: int = 60,
                 factor: float = 252.0,
                 transaction_cost: float = 0.001,
                 min_weight: float = 0.0,
                 max_weight: float = 1.0):
        self.indexDict = indexDict
        self.priceDict = priceDict
        self.volumeDict = volumeDict
        self.indexWeightCol = indexWeightCol
        self.indexMemberCol = indexMemberCol
        self.returnsColumn = returnsColumn
        self.volumeColumn = volumeColumn
        self.FeatEngInst = IndexFeatureEngineering(
                                              indexDict = indexDict,
                                              trading_days = trading_days,
                                              priceDict = priceDict,
                                              volumeDict = volumeDict,
                                              numberOfMembers = numberOfMembers,
                                              indexWeightCol = indexWeightCol,
                                              indexMemberCol = indexMemberCol,
                                              returnsColumn = returnsColumn,
                                              volumeColumn = volumeColumn,
                                              getLogReturns = getLogReturns,
                                              logTransformVolume = logVolume,
                                              standardizeVolume = standardizeVolume)
        self.dates = trading_days
        self.T = self.FeatEngInst.T
        self.K = lookbackWindow
        self.transaction_cost = transaction_cost
        self.N = numberOfMembers
        self.cost_function = CostFunctions(transaction_cost = self.transaction_cost) 
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.factor = factor

    def getStrategyWeights(self,
                        methods: list = ['HIS', 'GS1', 'GS2', 'LW'],
                        thresholdvalues: dict = {'GS1': [{'returns' : 0.5, 'volume': 0.5}], 'GS2' : [{'returns' : 0.5, 'volume': 0.5}]}):
        self.methods = []
        if 'GS1' in methods:
            for tDict in thresholdvalues['GS1']:
                self.methods += ['GS1' + '_' + str(tDict['returns']) + '_' + str(tDict['volume'])]
        if 'GS2' in methods:
            for tDict in thresholdvalues['GS2']:
                self.methods += ['GS2' + '_' + str(tDict['returns']) + '_' + str(tDict['volume'])]
        if 'HIS' in methods:
            self.methods += ['HIS']
        if 'LW' in methods:
            self.methods += ['LW']
        if 'IDN' in methods:
            self.methods += ['IDN']
        firstWeight = {asset : 0 for asset in self.FeatEngInst.getReturns(1, self.K + 1).columns}
        self.weights = {self.K + 1 : {method: firstWeight for method in self.methods}}
        for t in tqdm(range(self.K + 1, self.T - 1), position = 0, leave = True):
            returnsWindow = self.FeatEngInst.getReturns(t - self.K, t)
            volumeWindow = self.FeatEngInst.getVolume(t - self.K, t)
            current_assets = list(returnsWindow.columns) # what shares are traded in this period
            prev_weights = self.weights[t]
            priceVolumeCovarianceMatrices = self.covarianceMatrixMethods(returnsMatrix = returnsWindow, 
                                                                         volumeMatrix = volumeWindow,
                                                                         methods = self.methods, 
                                                                         thresholdvalues = thresholdvalues)
            t1 = {}
            for method in list(priceVolumeCovarianceMatrices.keys()):
                self.PortOptInst = IndexPortfolioOptimisation(cost_function = self.cost_function, 
                                                            min_weight = self.min_weight, 
                                                            max_weight = self.max_weight, 
                                                            factor = self.factor)
                w = self.PortOptInst.minimum_variance(covMatrix = priceVolumeCovarianceMatrices[method], 
                                                                    prev_weights = prev_weights[method]).fit()
                t1[method] = dict(zip(current_assets,w))
            self.weights[t+1] = t1
        return self
    

    def backtestStrategy(self,
                         AUM: float = 1e6):
        self.simpleReturns = IndexFeatureEngineering(
                                              indexDict = self.indexDict,
                                              trading_days = self.dates,
                                              priceDict = self.priceDict,
                                              volumeDict = self.volumeDict,
                                              numberOfMembers = self.N,
                                              indexWeightCol = self.indexWeightCol,
                                              indexMemberCol = self.indexMemberCol,
                                              returnsColumn = self.returnsColumn,
                                              volumeColumn = self.volumeColumn,
                                              getLogReturns = False,
                                              logTransformVolume = False,
                                              standardizeVolume = False)
        self.backtestResults = {t : {method: {} for method in self.methods} for t in range(self.K+1, self.T)}
        self.backtestResults[self.K+1] = {method: {'Portfolio Returns' : 0, 
                                                   'Portfolio Value' : AUM,
                                                   'Volume Bought' : 0,
                                                   'Volume Sold' : 0} for method in self.methods}
        for t in tqdm(range(self.K + 2, self.T), position = 0, leave = True):
            for method in self.methods:
                prev_weights = self.weights[t-1][method]
                weights = self.weights[t][method] # calculated on data up to and including t-1; traded OOS at time t
                current_assets = list(weights.keys())
                simple_returns = self.simpleReturns.getReturns(t, t+1, assets = current_assets).copy() # returns at time t
                simple_returns = (simple_returns + 1).prod()-1
                prev_portfolio_value = self.backtestResults[t-1][method]['Portfolio Value']
                tPortfolioReturns, tPortfolioValue, volBought, volSold = self.calculatePortfolioValue(simple_returns,
                                                                                                    weights,
                                                                                                    prev_weights,
                                                                                                    prev_portfolio_value)
                self.backtestResults[t][method] = {'Portfolio Returns' : tPortfolioReturns, 
                                                   'Portfolio Value' : tPortfolioValue,
                                                   'Volume Bought' : volBought,
                                                   'Volume Sold': volSold}
        return self.backtestResults


    def covarianceMatrixMethods(self, 
                                returnsMatrix,
                                volumeMatrix,
                                methods: list = ['HIS','GS1','GS2', 'LW'],
                                thresholdvalues: dict = {'GS1' : [{'returns' : 1.0, 'volume': 1.0}], 
                                                         'GS2' : [{'returns' : 1.0, 'volume': 1.0}]}):
        covMatInst = CovarianceMatrixVolume(returnsMatrix = returnsMatrix,
                                            volumeMatrix = volumeMatrix)
        matrices = {}
        for method in methods:
            if 'GS1' == method[:3]:
                for tDict in thresholdvalues['GS1']:
                    key = 'GS1' + '_' + str(tDict['returns']) + '_' + str(tDict['volume'])
                    returnsThreshold = ThresholdVectors(returnsMatrix).standard_deviation(c = tDict['returns'])
                    volumeThreshold = ThresholdVectors(volumeMatrix).standard_deviation(c = tDict['volume'])
                    matrices[key] = covMatInst.GerberMatrix1(returnsThreshold, volumeThreshold, method = 'quick')[1]
            if 'GS2' == method[:3]:
                for tDict in thresholdvalues['GS2']:
                    key = 'GS2' + '_' + str(tDict['returns']) + '_' + str(tDict['volume'])
                    returnsThreshold = ThresholdVectors(returnsMatrix).standard_deviation(c = tDict['returns'])
                    volumeThreshold = ThresholdVectors(volumeMatrix).standard_deviation(c = tDict['volume'])
                    matrices[key] = covMatInst.GerberMatrix2(returnsThreshold, volumeThreshold, method = 'quick')[1]
            if method == 'HIS':
                matrices['HIS'] = covMatInst.historicalCovariance()[1]
            if method == 'LW':
                matrices['LW'] = covMatInst.ledoitWolf()[1]
            if method == 'IDN':
                matrices['IDN'] = covMatInst.identity()[1]
        return matrices
    
    def calculatePortfolioValue(self,
                                simple_returns: pd.DataFrame,
                                weights: np.ndarray,
                                prev_weights: np.ndarray,
                                prev_portfolio_value: float):
        current_members = list(weights.keys())
        volumesTraded = self.calculateVolumesTraded(weights, prev_weights)
        volBought = volumesTraded['bought']*prev_portfolio_value
        volSold = volumesTraded['sold']*prev_portfolio_value
        costToTrade = (volBought + volSold) * self.transaction_cost
        weightedReturns = np.sum([weights[member] * simple_returns[member] for member in current_members])
        portfolioReturns = weightedReturns * (prev_portfolio_value  - costToTrade) # dollar value
        portfolioValue = portfolioReturns + prev_portfolio_value 
        return portfolioReturns, portfolioValue, volBought, volSold
    
    def calculateVolumesTraded(self,
                               weights: np.ndarray,
                               prev_weights: np.ndarray):
        current_members = list(weights.keys()) 
        prev_members = list(prev_weights.keys())
        exited_weights = np.array([prev_weights[member] for member in prev_members if member not in current_members])
        new_weights = np.array([weights[member] for member in current_members if member not in prev_members])
        consistent_members = [member for member in current_members if member in prev_members]
        consistent_weights_change = np.array([weights[member] - prev_weights[member] for member in consistent_members])

        volumeBought  = np.sum([i for i in consistent_weights_change if i > 0])
        volumeBought += np.sum([i for i in new_weights if i > 0])
        volumeBought += np.sum([i for i in -1*exited_weights if i > 0])

        volumeSold  = np.sum([i for i in consistent_weights_change if i < 0])
        volumeSold += np.sum([i for i in new_weights if i < 0])
        volumeSold += np.sum([i for i in -1*exited_weights if i < 0])
        return {'bought' : volumeBought, 'sold': np.abs(volumeSold)}