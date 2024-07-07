import datetime
import time


# 1d 기준 endTime 계산
def cal_end_time() -> int:
    now = datetime.datetime.now(datetime.UTC)
    ent_time = (
        now.replace(hour=0, minute=0, second=0, microsecond=0)
        - datetime.timedelta(days=1)
        + datetime.timedelta(milliseconds=100)
    )

    ent_time = int(ent_time.timestamp() * 1000)

    return ent_time


# 1d 기준 다음 기간
def wait_next_day():
    now = datetime.datetime.now(datetime.UTC)

    next_time = (
        now.replace(hour=0, minute=0, second=0, microsecond=0)
        + datetime.timedelta(days=1)
        + datetime.timedelta(milliseconds=100)
    )

    wait_seconds = (next_time - now).total_seconds()
    time.sleep(wait_seconds)
