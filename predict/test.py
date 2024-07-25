import pandas as pd

# 예제 데이터프레임 생성
data = {
    "A": [1, 2, 3],
    "B": [4, 5, 6],
    "C": [7, 8, 9],
    "D": [10, 11, 12],
    "E": [13, 14, 15],
}

df = pd.DataFrame(data)


print(df[["D", "B", "A"]])
