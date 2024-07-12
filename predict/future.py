import config
import datetime
import pandas as pd
from fetch import fetch_data, fetch_one_data, fetch_interval_data
from calculate import start_calculate_euclidean, procedure_calculate_euclidean
from process import calculate_values, process_data


def main(numbers: int):
    symbol = config.symbol

    data_4h = fetch_data(symbol=symbol, interval="4h", numbers=numbers, type="future")
    results_4h = start_calculate_euclidean(data_4h)
    results_1h = procedure_calculate_euclidean(results_4h, "1h", "future")
    if not results_1h[1]:
        return "No result"
    results_15m = procedure_calculate_euclidean(results_1h, "15m", "future")
    if not results_15m[1]:
        return "No result"
    results_5m = procedure_calculate_euclidean(results_15m, "5m", "future")
    if not results_5m[1]:
        return "No result"
    results_1m = procedure_calculate_euclidean(results_5m, "1m", "future")
    if not results_1m[1]:
        return "No result"

    end_time_cal = int(results_1m[1][0].iloc[-1]["close_time"])
    final_4h_cal = fetch_one_data(
        symbol=symbol, interval="4h", end_time=end_time_cal, limit=50, type="future"
    )

    start_time = int(results_1m[1][0].iloc[-1]["close_time"] + 1)
    end_time = int(start_time + datetime.timedelta(hours=24).total_seconds() * 1000 - 1)

    final_4h = fetch_interval_data(
        symbol=symbol,
        interval="4h",
        start_time=start_time,
        end_time=end_time,
        type="future",
    )
    final_4h = pd.concat([final_4h_cal, final_4h], axis=0, ignore_index=True)
    final_4h = calculate_values(final_4h)
    final_4h = process_data(final_4h)

    print(final_4h[["delta_process"]])
    print(final_4h["delta_process"].sum())


if __name__ == "__main__":
    main(20000)
