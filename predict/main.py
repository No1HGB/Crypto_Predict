import config
from fetch import fetch_data, fetch_interval_data
from process import calculate_values, process_data

symbol = config.symbol

data = fetch_data(symbol=symbol, interval="4h", numbers=20000)
data = calculate_values(data)
data = process_data(data)
print(data)
