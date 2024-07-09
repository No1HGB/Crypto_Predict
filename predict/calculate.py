import time
from typing import List, Tuple
import numpy as np
import pandas as pd
from config import symbol
from fetch import fetch_one_data, fetch_interval_data
from process import calculate_values, process_data


# 4시간 봉 6개 데이터 유클리드 거리 계산
def start_calculate_euclidean(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[pd.DataFrame]]:

    df = calculate_values(df)
    df = process_data(df)

    standard_rows = df.iloc[-6:]

    data = df[["volume_process", "delta_process", "distance_process"]]
    # 마지막 n개 행 데이터
    last_rows = data.iloc[-6:].values

    # 모든 거리 계산을 위해 데이터 배열 준비
    data_matrix = data.iloc[:-6].values

    # 유클리드 거리 계산
    distances = []
    for i in range(len(data_matrix) - 5):
        subset = data_matrix[i : i + 6]
        dist = np.sum(np.linalg.norm(last_rows - subset, axis=1))
        distances.append((i, dist))

    # 거리가 가장 작은 순서대로 정렬
    distances_sorted = sorted(distances, key=lambda x: x[1])

    # 거리가 가장 작은 데이터셋 16개 반환
    closest_datasets = [df.iloc[i : i + 6] for i, _ in distances_sorted[:16]]

    return standard_rows, closest_datasets


# 1h,15m,5m,1m 데이터 유클리드 거리 계산
# results 데이터보다 한 depth 아래 interval 대입
def procedure_calculate_euclidean(
    results: Tuple[pd.DataFrame, List[pd.DataFrame]], interval: str
) -> Tuple[pd.DataFrame, List[pd.DataFrame]]:

    # 데이터 정의
    standard_rows_init = results[0]
    closest_datasets_init = results[1]

    # 50일 계산용 데이터 가져오기
    end_time_cal = int(standard_rows_init.iloc[0]["open_time"] - 1)
    df_for_cal = fetch_one_data(
        symbol=symbol, interval=interval, end_time=end_time_cal, limit=50
    )

    # 기준 데이터 가져오기
    standard_start_time = int(standard_rows_init.iloc[0]["open_time"])
    standard_end_time = int(standard_rows_init.iloc[-1]["close_time"])

    standard_df = fetch_interval_data(
        symbol=symbol,
        interval=interval,
        start_time=standard_start_time,
        end_time=standard_end_time,
    )
    standard_df = pd.concat([df_for_cal, standard_df])
    print(standard_df)
    standard_df = calculate_values(standard_df)
    standard_df = process_data(standard_df)
    print(standard_df)

    # 유클리드 거리 계산
    distances = []
    standard_data = standard_df[
        ["volume_process", "delta_process", "distance_process"]
    ].values

    time.sleep(0.1)

    for i in range(len(closest_datasets_init)):
        end_time_cal_com = int(closest_datasets_init[i].iloc[0]["open_time"] - 1)
        df_for_cal_com = fetch_one_data(
            symbol=symbol, interval=interval, end_time=end_time_cal_com, limit=50
        )

        comparison_start_time = int(closest_datasets_init[i].iloc[0]["open_time"])
        comparison_end_time = int(closest_datasets_init[i].iloc[-1]["close_time"])
        comparison_df = fetch_interval_data(
            symbol=symbol,
            interval=interval,
            start_time=comparison_start_time,
            end_time=comparison_end_time,
        )
        comparison_df = pd.concat([df_for_cal_com, comparison_df])
        print(comparison_df)
        comparison_df = calculate_values(comparison_df)
        comparison_df = process_data(comparison_df)
        print(comparison_df)

        comparison_data = comparison_df[
            ["volume_process", "delta_process", "distance_process"]
        ].values
        dist = np.sum(np.linalg.norm(standard_data - comparison_data, axis=1))
        distances.append((dist, comparison_df))
        time.sleep(0.1)

    distances_sorted = sorted(distances, key=lambda x: x[0])

    num: int = 0
    if interval == "1h":
        num = 8
    elif interval == "15m":
        num = 4
    elif interval == "5m":
        num = 2
    elif interval == "1m":
        num = 1

    closest_datasets = [df for _, df in distances_sorted[:num]]

    return standard_df, closest_datasets
