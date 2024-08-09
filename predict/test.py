import pandas as pd

data_chunk = pd.read_csv(
    "data/conv2d.csv",
    usecols=["open", "close", "high", "low", "volume"],
    header=0,
    skiprows=list(range(1, 7)),
    nrows=3,
)

print(data_chunk)
