import datetime
from fetch import fetch_one_data


startDate_str = input()
endDate_str = input()
startDate_time_obj = datetime.datetime.strptime(startDate_str, "%y-%m-%d %H:%M")
endDate_time_obj = datetime.datetime.strptime(endDate_str, "%y-%m-%d %H:%M")
# Unix 타임스탬프를 초 단위로 변환 후 밀리초로 변환
start_time = int(startDate_time_obj.timestamp() * 1000)
end_time = int(endDate_time_obj.timestamp() * 1000)

data = fetch_one_data(symbol="BTCUSDT", interval="4h", end_time=end_time, limit=100)
print(data)
