import yfinance as yf
import pandas as pd


def get_historical_data(
    ticker: str,
    period: str = "2y",
    start_date: str = None,
    end_date: str = None,
) -> pd.DataFrame | None:
    """
    Downloads historical OHLCV data from Yahoo Finance.
    If start_date and end_date are provided, they override the period parameter.
    """
    try:
        if start_date and end_date:
            data = yf.download(
                ticker, start=start_date, end=end_date, interval="1d", progress=False
            )
        else:
            data = yf.download(ticker, period=period, interval="1d", progress=False)

        if data.empty:
            return None

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        if data.index.name != "Date":
            data.index.name = "Date"

        data.reset_index(inplace=True)

        if "Date" not in data.columns and "index" in data.columns:
            data.rename(columns={"index": "Date"}, inplace=True)

        return data
    except Exception as e:
        print(f"Error downloading data for {ticker}: {e}")
        return None


def get_portfolio_data(
    tickers: list,
    period: str = "2y",
    start_date: str = None,
    end_date: str = None,
) -> dict:
    """Downloads OHLCV data for an entire portfolio."""
    portfolio_data = {}
    for ticker in tickers:
        df = get_historical_data(
            ticker, period=period, start_date=start_date, end_date=end_date
        )
        if df is not None:
            portfolio_data[ticker] = df
    return portfolio_data
