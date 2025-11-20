import pandas as pd
import yfinance as yf

class YahooDownloader:
    """
    Class to download stock data from Yahoo Finance.

    Attributes
    ----------
        start_date : str
            start date of the data 
        end_date : str
            end date of the data
        ticker_list : list
            a list of stock tickers to get info from

    Methods
    -------
    fetch_data()
        Fetches data from yahoo API

    """

    def __init__(self, start_date: str, end_date: str, ticker_list: list):
        self.start_date = start_date
        self.end_date = end_date
        self.ticker_list = ticker_list

    def fetch_data(self, auto_adjust=True) -> pd.DataFrame:
        """
        Fetches data from Yahoo API
        
        Parameters
        ----------
            auto_adjust: bool
                Adjust all OHLC automatically to avoid price gaps resulting from splits or dividends. 

        Returns
        -------
        `pd.DataFrame`
            7 columns: A date, open, high, low, close, volume and tick symbol
            for the specified stock ticker.

        """

        # Download and save the data in a pandas DataFrame:
        data_df = pd.DataFrame()
        num_failures = 0
        for tic in self.ticker_list:
            temp_df = yf.download(
                tic,
                start=self.start_date,
                end=self.end_date,
                auto_adjust=auto_adjust,
            )
            if temp_df.columns.nlevels != 1:
                temp_df.columns = temp_df.columns.droplevel(1)
            temp_df["tic"] = tic
            if len(temp_df) > 0:
                data_df = pd.concat([data_df, temp_df], axis=0)
            else:
                num_failures += 1
        if num_failures == len(self.ticker_list):
            raise ValueError("no data is fetched.")

        data_df = data_df.reset_index()
        
        try:
            # Convert column names 
            data_df.rename(
                columns={
                    "Date": "date",
                    "Adj Close": "adjcp",
                    "Close": "close",
                    "High": "high",
                    "Low": "low",
                    "Volume": "volume",
                    "Open": "open",
                    "tic": "tic",
                },
                inplace=True,
            )

        except NotImplementedError:
            print("the features are not supported currently")
        
        # Create day of the week column (monday = 0)
        data_df["day"] = data_df["date"].dt.dayofweek
        
        # Convert date to standard string format
        data_df["date"] = data_df.date.apply(lambda x: x.strftime("%Y-%m-%d"))
        
        # Drop missing data
        data_df = data_df.dropna()
        data_df = data_df.reset_index(drop=True)
        print("Shape of DataFrame: ", data_df.shape)

        data_df = data_df.sort_values(by=["date", "tic"]).reset_index(drop=True)

        return data_df