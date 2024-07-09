import config
import datetime
from fetch import fetch_data, fetch_interval_data
from calculate import start_calculate_euclidean, procedure_calculate_euclidean
from process import calculate_values, process_data


def main(numbers: int):
    symbol = config.symbol

    data_4h = fetch_data(symbol=symbol, interval="4h", numbers=numbers)
    results_4h = start_calculate_euclidean(data_4h)
    results_1h = procedure_calculate_euclidean(results_4h, "1h")
    results_15m = procedure_calculate_euclidean(results_1h, "15m")
    results_5m = procedure_calculate_euclidean(results_15m, "5m")
    results_1m = procedure_calculate_euclidean(results_5m, "1m")

    start_time = results_1m[1][0].iloc[0]["open_time"] + int(
        datetime.timedelta(hours=4).total_seconds() * 1000
    )
    end_time = results_1m[1][0].iloc[-1]["close_time"] + int(
        datetime.timedelta(hours=4).total_seconds() * 1000
    )
    final_4h = fetch_interval_data(
        symbol=symbol, interval="4h", start_time=start_time, end_time=end_time
    )
    final_4h = calculate_values(final_4h)
    final_4h = process_data(final_4h)
    print(final_4h[["delta_process"]])
    print(final_4h["delta_process"].sum())


if __name__ == "__main__":
    main(20000)
