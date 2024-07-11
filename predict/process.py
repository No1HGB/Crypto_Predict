import pandas as pd


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
    df = df.iloc[50:]
    df.reset_index(drop=True, inplace=True)

    return df
