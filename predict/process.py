import numpy as np
import pandas as pd


def cal_value(df: pd.DataFrame) -> pd.DataFrame:
    df["MA50"] = df["close"].rolling(window=50).mean()
    df["MA200"] = df["close"].rolling(window=200).mean()
    df["volume_MA50"] = df["volume"].rolling(window=50).mean()
    df["volume_MA200"] = df["volume"].rolling(window=200).mean()
    df.dropna(axis=0, inplace=True, how="any")

    # 0 값을 가지는 행 제거
    df = df[(df["open"] > 0) & (df["close"] > 0) & (df["volume"] > 0)].copy()

    df["delta"] = df["close"] / df["open"]
    df["up_delta"] = df["high"] / df[["open", "close"]].max(axis=1)
    df["down_delta"] = df["low"] / df[["open", "close"]].min(axis=1)
    df["d50"] = df["close"] / df["MA50"]
    df["d200"] = df["close"] / df["MA200"]

    df["volume_delta"] = df["volume"] / df["volume"].shift(1)
    df["volume_d50"] = df["volume"] / df["volume_MA50"]
    df["volume_d200"] = df["volume"] / df["volume_MA200"]

    df.drop(["MA50", "MA200", "volume_MA50", "volume_MA200"], axis=1, inplace=True)
    df.dropna(axis=0, inplace=True, how="any")
    df.reset_index(drop=True, inplace=True)

    return df


def create_x_data(future: pd.DataFrame, window_size=864):
    future_val = future[["up_delta", "delta", "down_delta", "volume_ratio"]]

    x_data = []
    for i in range(len(future_val) - 2 * window_size + 1):
        future_slice = future_val.iloc[i : i + window_size].values
        x_data.append(future_slice)
    return np.array(x_data)


def create_y_data(future: pd.DataFrame, window_size=864):
    future_val = future[["up_delta", "delta", "down_delta", "volume_ratio"]]

    y_data = []
    for i in range(window_size, len(future) - window_size + 1):
        day_num = int(window_size / 3)
        future_slice = future_val.iloc[i : i + day_num]
        delta_vector = future_slice["delta"].values
        y_data.append(delta_vector)
    return np.array(y_data)


def create_x_data_conv2d(future: pd.DataFrame, x_cols: list, x_days: int, y_days: int):
    window_size = x_days * 24 * 12
    y_window_size = y_days * 24 * 12
    future_val = future[x_cols]
    x_data = []

    for i in range(len(future_val) - (window_size + y_window_size) + 1):
        future_slice = future_val.iloc[i : i + window_size]
        slice_vectors = future_slice.values
        x_data.append(slice_vectors)

    return np.array(x_data)


def create_y_data_conv2d(future: pd.DataFrame, y_cols: list, x_days: int, y_days: int):
    window_size = x_days * 24 * 12
    y_window_size = y_days * 24 * 12
    future_val = future[y_cols]
    y_data = []

    for i in range(window_size, len(future) - y_window_size + 1):
        future_slice = future_val.iloc[i : i + y_window_size]
        slice_vectors = future_slice.values
        y_data.append(slice_vectors)

    return np.array(y_data)


# 메모리 효율화를 위한 x,y data
def generate_x_data_conv2d(
    future: pd.DataFrame, x_cols: list, x_days: int, y_days: int
):
    window_size = x_days * 24 * 12
    future_val = future[x_cols].values.astype(np.float32)

    for i in range(len(future_val) - window_size - y_days * 24 * 12 + 1):
        yield future_val[i : i + window_size]


def generate_y_data_conv2d(
    future: pd.DataFrame, y_cols: list, x_days: int, y_days: int
):
    window_size = x_days * 24 * 12  # x_days 만큼의 데이터 크기
    future_val = future[y_cols].values.astype(np.float32)

    for i in range(window_size, len(future_val) - y_days * 24 * 12 + 1):
        window = future_val[i : i + y_days * 24 * 12]

        open_first = window[0][0]  # 첫 번째 open 값
        close_last = window[-1][1]  # 마지막 close 값
        high_max = window[:, 2].max()  # high의 최댓값
        low_min = window[:, 3].min()  # low의 최솟값

        min_open_close = min(open_first, close_last)
        max_open_close = max(open_first, close_last)

        low_ratio = low_min / min_open_close
        close_open_ratio = close_last / open_first
        high_ratio = high_max / max_open_close

        yield [low_ratio, close_open_ratio, high_ratio]
