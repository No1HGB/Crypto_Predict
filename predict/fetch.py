from binance.spot import Spot
import pandas as pd
import datetime


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
            "taker_buy_quote_asset_volume",
            "ignore",
        ],
        axis=1,
        inplace=True,
    )
    df.rename(columns={"taker_buy_base_asset_volume": "taker_buy"}, inplace=True)

    # 모든 열을 숫자로 변환
    for column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    return df


def fetch_data(symbol: str, numbers: int, interval="1d") -> pd.DataFrame:
    now = datetime.datetime.now(datetime.UTC)
    end_datetime = now.replace(
        hour=0, minute=0, second=0, microsecond=0
    ) - datetime.timedelta(days=1)

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
        data.insert(0, df)
        end_time -= int(datetime.timedelta(days=num).total_seconds() * 1000)
        numbers -= num

    data_combined = pd.concat(data)
    data_combined.reset_index(drop=True, inplace=True)

    return data_combined
