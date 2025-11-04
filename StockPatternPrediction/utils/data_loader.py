import yfinance as yf
import pandas as pd

def fetch_stock_data(ticker, period="5y", interval="1d"):
    data = yf.download(ticker, period=period, interval=interval, progress=False)
    data.reset_index(inplace=True)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] if col[1] == '' or col[1] == ticker else col[0] for col in data.columns]
    data.columns = [str(col) for col in data.columns]
    rename_dict = {}
    for col in data.columns:
        key = col.lower()
        if key in ['open', 'high', 'low', 'close', 'date', 'datetime']:
            rename_dict[col] = key.capitalize()
        if key == "adj close" and "Close" not in data.columns:
            rename_dict[col] = "Close"
    data.rename(columns=rename_dict, inplace=True)
    if "Date" not in data.columns and "Datetime" in data.columns:
        data["Date"] = pd.to_datetime(data["Datetime"])
    elif "Date" in data.columns:
        data["Date"] = pd.to_datetime(data["Date"])
    ohlc_cols = ["Open", "High", "Low", "Close"]
    missing_cols = [col for col in ohlc_cols if col not in data.columns]
    if missing_cols:
        raise KeyError(f"Missing required columns in data: {missing_cols}. Columns found: {list(data.columns)}")
    data = data.dropna(subset=ohlc_cols)
    return data
