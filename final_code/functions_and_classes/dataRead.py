from glob import glob
import pandas as pd

##########################
## NOT TO BE USED ########
## REFERENCE ONLY ########
##########################

def readRawDataYF(tickers: list,
                startDate: str,
                endDate: str,
                path = '../../data/yahoo_finance'):
    globFiles = {ticker: glob(f'{path}/data_{ticker}_{startDate}_{endDate}.pickle')[0] for ticker in tickers}
    data = {ticker: pd.read_pickle(globFiles[ticker]) for ticker in tickers}
    return data

def readRawDataBBG1(tickers: list,
                    start_date: str,
                    end_date: str,
                    date_column = 'date',
                    metric = 'PX_LAST',
                    frequency = 'MONTHLY',
                    path = '../../data/dataBBG/dataBBG'):
    start_date = pd.to_datetime(start_date).date()
    end_date = pd.to_datetime(end_date).date()
    globFiles = {ticker: glob(f'{path}/data-{ticker}-{metric}_{frequency}.pickle')[0] for ticker in tickers}
    data = {}
    for ticker in tickers:
        df = pd.read_pickle(globFiles[ticker]).set_index(date_column)
        df = df.loc[start_date:end_date]
        data[ticker] = df
    return data