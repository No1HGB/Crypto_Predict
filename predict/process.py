import numpy as np
import pandas as pd


def cal_value(df: pd.DataFrame) -> pd.DataFrame:
    alpha = 2 / (50 + 1)
    alpha_200 = 2 / (200 + 1)
    df["EMA50"] = df["close"].ewm(alpha=alpha, adjust=True).mean()
    df["EMA200"] = df["close"].ewm(alpha=alpha_200, adjust=False).mean()
    df["volume_MA"] = df["volume"].rolling(window=50).mean()

    df["delta"] = df["close"] / df["open"]
    df["up_delta"] = df["high"] / df[["open", "close"]].max(axis=1)
    df["down_delta"] = df["low"] / df[["open", "close"]].min(axis=1)

    df["volume_ratio"] = df["volume"] / df["volume_MA"]
    df["volume_delta"] = df["volume"] / df["volume"].shift(1)

    df["distance_50"] = df["close"] / df["EMA50"]
    df["distance_200"] = df["close"] / df["EMA200"]

    df.drop(["EMA50", "EMA200", "volume_MA"], axis=1, inplace=True)
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

        # future_part = future_slice.iloc[:d_num]
        # future_open = future_part.iloc[0]["open"]
        # future_close = future_part.iloc[-1]["close"]
        #
        # a1 = (future_close - future_open) / future_open * 100
        # a2 = (
        #     (future_part["high"].max() - max(future_open, future_close))
        #     / max(future_open, future_close)
        #     * 100
        # )
        # a3 = (
        #     (future_part["low"].min() - min(future_open, future_close))
        #     / min(future_open, future_close)
        #     * 100
        # )
        #
        # y_data.append([a1, a2, a3])
    return np.array(y_data)


# 열 별 MinMaxScaling
def min_max_scaling(df: pd.DataFrame):
    min_max_values = {}
    df_scaled = df.copy()
    for column in df.columns:
        min_value = df[column].min()
        max_value = df[column].max()
        df_scaled[column] = (df[column] - min_value) / (max_value - min_value)
        min_max_values[column] = (min_value, max_value)
    return df_scaled, min_max_values


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


# y_data 역 스케일링
def inverse_scaling(scaled_data: pd.DataFrame, min_max_values_list: list, y_cols: list):
    original_data = []

    for i in range(len(scaled_data)):
        # 각 슬라이스를 (timesteps, features, 1)에서 (timesteps, features)로 reshape
        scaled_slice = scaled_data[i].reshape(-1, len(y_cols))
        min_max_values = min_max_values_list[i]
        original_slice = pd.DataFrame(scaled_slice, columns=y_cols)

        for column in original_slice.columns:
            min_value, max_value = min_max_values[column]
            original_slice[column] = (
                original_slice[column] * (max_value - min_value) + min_value
            )

        original_data.append(original_slice.values)

    return np.array(original_data)


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
    window_size = x_days * 24 * 12
    future_val = future[y_cols].values.astype(np.float32)

    for i in range(window_size, len(future_val) - y_days * 24 * 12 + 1):
        yield future_val[i : i + y_days * 24 * 12]
