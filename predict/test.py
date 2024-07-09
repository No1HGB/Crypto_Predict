import datetime
import pandas as pd

"""
startDate_str = input()
endDate_str = input()
startDate_time_obj = datetime.datetime.strptime(startDate_str, "%y-%m-%d %H:%M")
endDate_time_obj = datetime.datetime.strptime(endDate_str, "%y-%m-%d %H:%M")
# Unix 타임스탬프를 초 단위로 변환 후 밀리초로 변환
start_time = int(startDate_time_obj.timestamp() * 1000)
end_time = int(endDate_time_obj.timestamp() * 1000)

data = fetch_one_data(symbol="BTCUSDT", interval="4h", end_time=end_time, limit=100)
print(data)
"""
df1 = pd.DataFrame(
    {
        "A": ["A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9"],
        "B": ["B0", "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9"],
        "C": ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"],
        "D": ["D0", "D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9"],
    }
)

print(df1[["A", "B"]])
