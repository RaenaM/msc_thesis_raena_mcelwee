import numpy as np
import pandas as pd
from pyfinance import TSeries
from scipy import stats

class PerformanceAnalysis():

    metric_index = ['Annual. Returns',
                    'Annual. Stand. Dev',
                    'Annual. Skew',
                    'Annual. Kurtosis',
                    'Total Returns',
                    'Arith. Returns',
                    'Geom. Returns',
                    'Sharpe Ratio',
                    'Max. Drawdown',
                    'Annual. Turnover',
                    'VaR']

    def __init__(self,
                 portfolioReturns: pd.Series,
                 portfolioValue: pd.Series,
                 volBought: pd.Series,
                 volSold: pd.Series,
                 factor: float = 252.0):
        
        assert factor == 252.0 or factor == 12.0, "Can only handle monthly or daily frequency data."

        self.portfolioReturns = portfolioReturns
        self.portfolioValue = portfolioValue
        self.initAUM = self.portfolioValue.iloc[0]
        self.percReturns = self.portfolioValue.pct_change()
        self.logReturns = np.log(self.portfolioValue).diff().dropna()
        self.F = factor
        self.FText = 'M' if factor == 12.0 else 'D' # daily or monthly returns
        self.tsReturns = TSeries(self.percReturns, freq = self.FText)
        self.annualReturns = self.tsReturns.rollup('A')
        self.volBought = volBought
        self.volSold = volSold
        
    def annualiseReturns(self):
        nYears = self.annualReturns.shape[0]
        self.anlzdReturns = (1 + self.annualReturns).prod()**(1/nYears) - 1
        return self.anlzdReturns

    def annualisedSTD(self,
                      ddof: int = 1):
        self.aSTD = self.percReturns.std(ddof=ddof)*np.sqrt(self.F)
        return self.aSTD
    
    def totalReturn(self):
        self.totR = self.tsReturns.cuml_ret()
        return self.totR
    
    def arithmeticReturn(self):
        self.arithR = self.annualReturns.mean()
        return self.arithR
    
    def geometricReturn(self):
        self.geomR = self.annualReturns.geomean()
        return self.geomR

    def annualisedSkew(self):
        self.skew = self.logReturns.skew()
        return self.skew
    
    def annualisedKurt(self):
        self.kurt = stats.kurtosis((self.logReturns).to_list(), fisher=False)
        return self.kurt

    # def sharpeRatio(self, 
    #                 rf: float = 0.02,
    #                 ddof: int = 1):
    #     self.aSTD = self.annualisedSTD(ddof = ddof)
    #     self.anlzdReturns = self.annualiseReturns()
    #     self.SR = (self.anlzdReturns - rf)/self.aSTD
    #     return self.SR

    def sharpeRatio(self, 
                    rf: float = 0.02,
                    ddof: int = 1):
        stdReturns = self.percReturns.std(ddof=ddof)
        meanReturns = self.percReturns.mean()
        self.SR = np.sqrt(self.F)*(meanReturns - rf)/stdReturns
        return self.SR
    
    def maxDrawdown(self):
        self.MD = self.tsReturns.max_drawdown()
        return self.MD
    
    def annualisedTurnover(self):
        totalBought = self.volBought.sum()
        totalSold = self.volSold.sum()
        totalPurchases = max(totalBought, totalSold)
        multiplier = self.F / self.tsReturns.shape[0]
        averageAssets = self.portfolioValue.mean()
        self.AT = (totalPurchases / averageAssets) * multiplier
        return self.AT
    
    def monthlyVaR(self, 
                   level: float = 0.95):
        assert level > 0 and level < 1, "Level must be in the open interval (0,1)."
        a = 1 - level
        self.VaR = self.percReturns.quantile(a)
        return self.VaR
    
    def calcAllMetrics(self,
                    ddof: int = 1,
                    rf: float = 0.02,
                    VaRlevel: float = 0.95):
        self.annualiseReturns()
        self.annualisedSTD(ddof = ddof)
        self.totalReturn()
        self.arithmeticReturn()
        self.geometricReturn()
        self.annualisedSkew()
        self.annualisedKurt()
        self.sharpeRatio(rf = rf, ddof = ddof)
        self.maxDrawdown()
        self.annualisedTurnover()
        self.monthlyVaR(level = VaRlevel)
        return self
    
    def metricSeries(self,
                   ddof: int = 1,
                   rf: float = 0.02,
                   VaRlevel: float = 0.95):
        self.calcAllMetrics(ddof = ddof, rf = rf, VaRlevel = VaRlevel)
        results =  {'Annual. Returns' : self.anlzdReturns,
                    'Annual. Stand. Dev': self.aSTD,
                    'Annual. Skew' : self.skew,
                    'Annual. Kurtosis' : self.kurt,
                    'Total Returns' : self.totR,
                    'Arith. Returns' : self.arithR,
                    'Geom. Returns' : self.geomR,
                    'Sharpe Ratio' : self.SR,
                    'Max. Drawdown' : self.MD,
                    'Annual. Turnover': self.AT,
                    'VaR': self.VaR}
        series_results = pd.Series(results, index = self.metric_index)
        return series_results