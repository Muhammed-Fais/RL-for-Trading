import yfinance as yf

def get_price_data(symbol, start_date, end_date):
    """
    Download historical price data from Yahoo Finance.
    """
    df = yf.download(symbol, start=start_date, end=end_date)
    return df[['Close']].values