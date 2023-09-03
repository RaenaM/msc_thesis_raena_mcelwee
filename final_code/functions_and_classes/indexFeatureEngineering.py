import numpy as np
import pandas as pd
from itertools import compress

class IndexFeatureEngineering():

    """Gets the appropriate returns and volume matrices necessary for backtesting and weights measurement.
    """

    def __init__(self, 
                 indexDict: dict,
                 trading_days: pd.core.indexes.datetimes.DatetimeIndex,
                 priceDict: dict,
                 volumeDict: dict = None,
                 numberOfMembers: int = 10,
                 indexWeightCol: str = 'Weight',
                 indexMemberCol: str = 'INDX_MWEIGHT_HIST',
                 returnsColumn: str = 'PX_LAST',
                 volumeColumn: str = 'PX_VOLUME',
                 getLogReturns: bool = False,
                 logTransformVolume: bool = True,
                 standardizeVolume: bool = True):
        """

        Args:
            indexDict (dict): dictionary, with string dates for keys in format "YYYYMMDD", and pandas.DataFrame for values containing the index tickers and their respective weightings
            trading_days (pandas.core.indexes.datetimes.DatetimeIndex): a list or DatetimeIndex object storing the dates of the trading days for the portfolio. 
            priceDict (dict): dictionary keyed by tickers, with pandas.DataFrame values containing prices indexed by date.
            volumeDict (dict, optional): dictionary keyed by tickers, with pandas.DataFrame values containing trading volumes indexed by date. Defaults to None.
            numberOfMembers (int, optional): number of assets to consider for returns and volume matrices, by most heavily weighted. Defaults to 10.
            indexWeightCol (str, optional): column name for index weights in `indexDict` values. Defaults to 'Weight'.
            indexMemberCol (str, optional): column name for index member tickers in `indexDict` values. Defaults to 'INDX_MWEIGHT_HIST'.
            returnsColumn (str, optional):  column name for asset prices in `priceDict` values. Defaults to 'PX_LAST'.
            volumeColumn (str, optional): column name for trading volume in `volumeDict` values. Defaults to 'PX_VOLUME'.
            getLogReturns (bool, optional): if False, get simple returns. Defaults to False.
            logTransformVolume (bool, optional): if False, do not log-transform volume. Defaults to True.
            standardizeVolume (bool, optional): if False, do not standardise volume. Defaults to True.
        """
        if volumeDict is not None:
            assert priceDict.keys() == volumeDict.keys(), 'Some tickers are different between the price and volume matrices.'
        self.indexDict = indexDict
        self.trading_days = trading_days
        self.returnsColumn = returnsColumn
        self.volumeColumn = volumeColumn
        self.priceDict = priceDict 
        self.volumeDict = volumeDict 
        self.indexWeightCol = indexWeightCol
        self.indexMemberCol = indexMemberCol
        self.getLogReturns = getLogReturns
        self.logTransformVolume = logTransformVolume
        self.standardizeVolume = standardizeVolume
        self.T = len(self.trading_days)
        self.rebalDates = pd.to_datetime(list(self.indexDict.keys()))
        self.M = numberOfMembers

    def getReturns(self,
                   windowStart: int = 0,
                   windowEnd: int = None,
                   assets = None):
        if windowEnd is None:
            windowEnd = self.T
        returnsDFWindow = self.getWindowData(windowStart, windowEnd, 'price', assets = assets).copy()
        returnsDFWindow = self.log_returns(returnsDFWindow) if self.getLogReturns else self.simple_returns(returnsDFWindow)
        returnsDFWindow = returnsDFWindow.iloc[1:,]
        return returnsDFWindow
    
    def getVolume(self,
                  windowStart: int,
                  windowEnd: int,
                  assets = None):
        volumeDFWindow = self.getWindowData(windowStart, windowEnd, 'volume', assets = assets)
        volumeDFWindow = np.log(volumeDFWindow) if self.logTransformVolume else volumeDFWindow
        volumeDFWindow = volumeDFWindow.iloc[1:,]
        volumeDFWindow = self.standardizer(volumeDFWindow) if self.standardizeVolume else volumeDFWindow
        return volumeDFWindow
    
    def getWindowData(self,
                      windowStart: int = 0,
                      windowEnd: int = None,
                      dataField: str = 'price',
                      assets = None):
        windowStartDate = self.trading_days[windowStart - 1]
        windowEndDate = self.trading_days[windowEnd - 1]
        windowRebalanceMembers = self.getIndexMembers(windowEndDate) if assets is None else assets
        if dataField == 'price':
            memberPriceData = {member: self.priceDict[member].loc[windowStartDate:windowEndDate, self.returnsColumn] for member in windowRebalanceMembers}
            memberDF = pd.DataFrame(memberPriceData)
        elif dataField == 'volume':
            memberVolumeData = {member: self.volumeDict[member].loc[windowStartDate:windowEndDate, self.volumeColumn] for member in windowRebalanceMembers}
            memberDF = pd.DataFrame(memberVolumeData)
        return memberDF

    def getIndexMembers(self,
                        windowEndDate: int):
        windowRebalanceDate = np.max(list(compress(self.rebalDates, self.rebalDates < windowEndDate)))
        windowRebalanceDate = windowRebalanceDate.strftime('%Y%m%d')
        windowRebalanceMembers = self.indexDict[windowRebalanceDate].sort_values(by = self.indexWeightCol, ascending = False)
        windowRebalanceMembers = list(windowRebalanceMembers[self.indexMemberCol].iloc[0:self.M])
        return windowRebalanceMembers

    def simple_returns(self, priceColumn): 
        return priceColumn.pct_change()

    def log_returns(self, priceColumn):
        return np.log(priceColumn) - np.log(priceColumn.shift(1))
    
    def standardizer(self, df):
        return (df - df.mean(axis = 0))/df.std(axis = 0) # sample variance, ddof = 1