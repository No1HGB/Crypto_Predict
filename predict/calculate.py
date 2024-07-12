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
    comparison_rows = df.iloc[:-6]

    # 마지막 n개 행 데이터
    last_rows = standard_rows[
        ["volume_process", "delta_process", "distance_process"]
    ].values

    # 모든 거리 계산을 위해 데이터 배열 준비
    comparison_matrix = comparison_rows[
        ["volume_process", "delta_process", "distance_process"]
    ].values

    # 유클리드 거리 계산
    distances = []
    for i in range(len(comparison_matrix) - 5):
        subset = comparison_matrix[i : i + 6]
        dist = np.sum(np.linalg.norm(last_rows - subset, axis=1))
        distances.append((dist, comparison_rows.iloc[i : i + 6]))

    # 거리가 가장 작은 순서대로 정렬
    distances_sorted = sorted(distances, key=lambda x: x[0])

    # 거리가 가장 작은 데이터셋 16개 반환
    closest_datasets = [dataset for _, dataset in distances_sorted[:16]]

    for df in closest_datasets:
        print(f"closest: {df.shape}")

    return standard_rows, closest_datasets


# 1h,15m,5m,1m 데이터 유클리드 거리 계산
# results 데이터보다 한 depth 아래 interval 대입
def procedure_calculate_euclidean(
    results: Tuple[pd.DataFrame, List[pd.DataFrame]],
    interval: str,
    type: str = "spot",
) -> Tuple[pd.DataFrame, List[pd.DataFrame]]:

    # 데이터 정의
    standard_rows_init = results[0]
    closest_datasets_init = results[1]

    # 50일 계산용 데이터 가져오기
    end_time_cal = int(standard_rows_init.iloc[0]["open_time"] - 1)
    df_for_cal = fetch_one_data(
        symbol=symbol,
        interval=interval,
        end_time=end_time_cal,
        limit=50,
        type=type,
    )

    # 기준 데이터 가져오기
    standard_start_time = int(standard_rows_init.iloc[0]["open_time"])
    standard_end_time = int(standard_rows_init.iloc[-1]["close_time"])

    standard_df = fetch_interval_data(
        symbol=symbol,
        interval=interval,
        start_time=standard_start_time,
        end_time=standard_end_time,
        type=type,
    )
    standard_df = pd.concat([df_for_cal, standard_df], axis=0, ignore_index=True)
    standard_df = calculate_values(standard_df)
    standard_df = process_data(standard_df)

    # 유클리드 거리 계산
    distances = []
    standard_data = standard_df[
        ["volume_process", "delta_process", "distance_process"]
    ].values

    time.sleep(0.1)

    for i in range(len(closest_datasets_init)):
        end_time_cal_com = int(closest_datasets_init[i].iloc[0]["open_time"] - 1)
        df_for_cal_com = fetch_one_data(
            symbol=symbol,
            interval=interval,
            end_time=end_time_cal_com,
            limit=50,
            type=type,
        )

        comparison_start_time = int(closest_datasets_init[i].iloc[0]["open_time"])
        comparison_end_time = int(closest_datasets_init[i].iloc[-1]["close_time"])
        if interval == "1h":
            hour = (comparison_end_time - comparison_start_time) / (3600 * 1000)
            print(hour)

        comparison_df = fetch_interval_data(
            symbol=symbol,
            interval=interval,
            start_time=comparison_start_time,
            end_time=comparison_end_time,
            type=type,
        )

        comparison_df = pd.concat(
            [df_for_cal_com, comparison_df], axis=0, ignore_index=True
        )
        comparison_df = calculate_values(comparison_df)
        comparison_df = process_data(comparison_df)
        comparison_data = comparison_df[
            ["volume_process", "delta_process", "distance_process"]
        ].values
        if standard_data.shape == comparison_data.shape:
            dist = np.sum(np.linalg.norm(standard_data - comparison_data, axis=1))
            distances.append((dist, comparison_df))
        time.sleep(0.1)

    distances_sorted = sorted(distances, key=lambda x: x[0])

    num: int = 0
    if distances_sorted:
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
