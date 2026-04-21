import yfinance as yf
ticker = "NU"
data = yf.download(ticker, period="2y")
print("Columns:", data.columns)
print("Data head:\n", data.head())
