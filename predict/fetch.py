from binance.spot import Spot
import pandas as pd
import datetime


# fetch_data 를 위한 함수, 정해진 개수의 데이터 가져옴
def fetch_one_data(
    symbol: str, interval: str, end_time: int, limit: int
) -> pd.DataFrame:
    client = Spot()
    bars = client.klines(
        symbol=symbol, interval=interval, endTime=end_time, limit=limit
    )
    df = pd.DataFrame(
        bars,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume",
            "ignore",
        ],
    )
    df.drop(
        [
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume",
            "ignore",
        ],
        axis=1,
        inplace=True,
    )

    # 모든 열을 숫자로 변환
    for column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    return df


# interval 에 따른 정해진 개수의 데이터 가져오기
def fetch_data(symbol: str, interval: str, numbers: int) -> pd.DataFrame:
    start_timedelta = datetime.timedelta()
    if interval == "1m":
        start_timedelta = datetime.timedelta(minutes=1)
    elif interval == "5m":
        start_timedelta = datetime.timedelta(minutes=5)
    elif interval == "15m":
        start_timedelta = datetime.timedelta(minutes=15)
    elif interval == "1h":
        start_timedelta = datetime.timedelta(hours=1)
    elif interval == "4h":
        start_timedelta = datetime.timedelta(hours=4)
    elif interval == "1d":
        start_timedelta = datetime.timedelta(days=1)

    now = datetime.datetime.now(datetime.UTC)
    end_datetime = (
        now.replace(hour=0, minute=0, second=0, microsecond=0) - start_timedelta
    )

    end_time = int(end_datetime.timestamp() * 1000)
    data = []

    while numbers > 0:
        if numbers < 1000:
            num = numbers
        else:
            num = 1000

        df = fetch_one_data(
            symbol=symbol, interval=interval, end_time=end_time, limit=num
        )
        if df.empty:
            break

        data.insert(0, df)
        interval_timedelta = datetime.timedelta()
        if interval == "1m":
            interval_timedelta = datetime.timedelta(minutes=1) * num
        elif interval == "5m":
            interval_timedelta = datetime.timedelta(minutes=5) * num
        elif interval == "15m":
            interval_timedelta = datetime.timedelta(minutes=15) * num
        elif interval == "1h":
            interval_timedelta = datetime.timedelta(hours=1) * num
        elif interval == "4h":
            interval_timedelta = datetime.timedelta(hours=4) * num
        elif interval == "1d":
            interval_timedelta = datetime.timedelta(days=1) * num

        end_time -= int(interval_timedelta.total_seconds() * 1000)
        numbers -= num

    data_combined = pd.concat(data)
    data_combined.reset_index(drop=True, inplace=True)

    return data_combined


# 시작 시간과 마지막 시간 사이의 데이터를 가져오는 함수
def fetch_interval_data(
    symbol: str, interval: str, start_time: int, end_time: int
) -> pd.DataFrame:
    client = Spot()
    bars = client.klines(
        symbol=symbol,
        interval=interval,
        startTime=start_time,
        endTime=end_time,
    )
    df = pd.DataFrame(
        bars,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume",
            "ignore",
        ],
    )
    df.drop(
        [
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume",
            "ignore",
        ],
        axis=1,
        inplace=True,
    )

    # 모든 열을 숫자로 변환
    for column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    return df
