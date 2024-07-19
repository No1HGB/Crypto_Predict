import numpy as np
import pandas as pd


def cal_value(df: pd.DataFrame) -> pd.DataFrame:
    alpha = 2 / (50 + 1)
    df["EMA50"] = df["close"].ewm(alpha=alpha, adjust=True).mean()

    df["delta"] = (df["close"] - df["open"]) / df["open"] * 100
    df["up_delta"] = (
        (df["high"] - df[["open", "close"]].max(axis=1))
        / df[["open", "close"]].max(axis=1)
        * 100
    )
    df["down_delta"] = (
        (df["low"] - df[["open", "close"]].min(axis=1))
        / df[["open", "close"]].min(axis=1)
        * 100
    )
    df["volume_delta"] = df["volume"] / df["volume"].shift(1) - 1
    df["distance"] = (df["close"] - df["EMA50"]) / df["EMA50"] * 100

    df.dropna(axis=0, inplace=True, how="any")
    df.reset_index(drop=True, inplace=True)

    return df


def create_x_data(spot: pd.DataFrame, future: pd.DataFrame, window_size=864):
    if len(spot) != len(future):
        raise ValueError("A and B must have same length")

    spot_val = spot[["delta", "up_delta", "down_delta", "volume_delta", "distance"]]
    future_val = future[["delta", "up_delta", "down_delta", "volume_delta", "distance"]]

    x_data = []
    for i in range(len(spot_val) - window_size):
        spot_slice = spot_val.iloc[i : i + window_size].values
        future_slice = future_val.iloc[i : i + window_size].values
        combined = np.vstack((spot_slice, future_slice))
        x_data.append(combined)
    return np.array(x_data)


def create_y_data(spot: pd.DataFrame, future: pd.DataFrame, window_size=864):
    if len(spot) != len(future):
        raise ValueError("A and B must have same length")

    y_data = []
    for i in range(window_size, len(spot) - window_size, window_size):
        spot_slice = spot.iloc[i : i + window_size]
        future_slice = future.iloc[i : i + window_size]

        spot_open = spot_slice.iloc[0]["open"]
        spot_close = spot_slice.iloc[-1]["close"]
        future_open = future_slice.iloc[0]["open"]
        future_close = future_slice.iloc[-1]["close"]

        a1 = (spot_close - spot_open) / spot_open * 100
        a2 = (
            (spot_slice["high"].max() - max(spot_open, spot_close))
            / max(spot_open, spot_close)
            * 100
        )
        a3 = (
            (spot_slice["low"].min() - min(spot_open, spot_close))
            / min(spot_open, spot_close)
            * 100
        )

        b1 = (future_close - future_open) / future_open * 100
        b2 = (
            (future_slice["high"].max() - max(future_open, future_close))
            / max(future_open, future_close)
            * 100
        )
        b3 = (
            (future_slice["low"].min() - min(future_open, future_close))
            / min(future_open, future_close)
            * 100
        )

        y_data.append([a1, a2, a3, b1, b2, b3])
    return np.array(y_data)


# 필요한 값들 계산
def calculate_values(df: pd.DataFrame) -> pd.DataFrame:
    alpha = 2 / (50 + 1)

    df["volume_MA"] = df["volume"].rolling(window=50).mean()
    df["volume_ratio"] = df["volume"] / df["volume_MA"]

    df["delta"] = (df["close"] - df["open"]) / df["open"] * 100
    df["delta_EMA"] = abs(df["delta"]).ewm(alpha=alpha, adjust=True).mean()

    df["EMA50"] = df["close"].ewm(alpha=alpha, adjust=True).mean()
    df["distance"] = (df["close"] - df["EMA50"]) / df["EMA50"] * 100
    df["distance_EMA"] = abs(df["distance"]).ewm(alpha=alpha, adjust=True).mean()

    return df


# 전처리
# v_r' = v_r * (가장 최근 volume MA) / (현재 데이터 volume MA)
# d' = d * (가장 최근 절댓값 delta EMA) / (현재 데이터 절댓값 delta EMA)
# dis' = dis * (가장 최근 절댓값 distance EMA) / (현재 데이터 절댓값 distance EMA)
def process_data(df: pd.DataFrame) -> pd.DataFrame:
    volume_mark = df.iloc[-1]["volume_MA"]
    df["volume_process"] = df["volume_ratio"] * (volume_mark / df["volume_MA"])

    delta_mark = df.iloc[-1]["delta_EMA"]
    df["delta_process"] = df["delta"] * (delta_mark / df["delta_EMA"])

    distance_mark = df.iloc[-1]["distance_EMA"]
    df["distance_process"] = df["distance"] * (distance_mark / df["distance_EMA"])

    df.drop(
        [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "EMA50",
        ],
        inplace=True,
        axis=1,
    )
    df.dropna(axis=0, inplace=True, how="any")
    df.reset_index(drop=True, inplace=True)

    return df
