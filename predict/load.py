import numpy as np
import pandas as pd
from tensorflow import keras

from process import create_x_data_conv2d, create_y_data_conv2d

# Keras 모델 불러오기
model = keras.models.load_model("model/conv2d_2.keras")

# 변수 설정
data_cnt: int = 12000
test_cnt: int = 12
epochs: int = 1000
x_cols: list = ["up_delta", "delta", "down_delta", "volume_ratio"]
y_cols: list = ["up_delta", "delta", "down_delta"]
activation: str = "leaky_relu"

# 데이터 가져오기
df = pd.read_csv("data/conv2d.csv")
data = df.iloc[-data_cnt:]

# 데이터 전처리 및 생성
x_data = create_x_data_conv2d(data, x_cols, 3, 1)
y_data = create_y_data_conv2d(data, y_cols, 3, 1)

# 데이터를 Conv2D 입력에 맞게 4차원으로 변환
x_data = x_data.reshape((x_data.shape[0], x_data.shape[1], x_data.shape[2], 1))
y_data = y_data.reshape((y_data.shape[0], y_data.shape[1], y_data.shape[2], 1))
x_shape_input = (x_data.shape[1], x_data.shape[2], 1)
y_shape_input = (y_data.shape[1], y_data.shape[2], 1)

# 학습 데이터, 테스트 데이터 분리
x_data_learn = x_data[:-test_cnt]
y_data_learn = y_data[:-test_cnt]
x_data_test = x_data[-test_cnt:]
y_data_test = y_data[-test_cnt:]

# 예측(테스트 데이터)
y_result = model.predict(x_data_test)
y_test = y_data_test.reshape(
    (y_data_test.shape[0], y_data_test.shape[1], y_data_test.shape[2])
)
y_pred = y_result.reshape(
    (y_data_test.shape[0], y_data_test.shape[1], y_data_test.shape[2])
)

# 예측 결과 출력
test_results = []
for test_window in y_test:
    init_price = 10000
    ups = []
    downs = []

    for test in test_window:
        ups.append(init_price * test[0])
        downs.append(init_price * test[2])
        init_price *= test[1]
    test_results.append([max(ups), init_price, min(downs)])

pred_results = []
for pred_window in y_pred:
    init_price = 10000
    ups = []
    downs = []
    for pred in pred_window:
        ups.append(init_price * pred[0])
        downs.append(init_price * pred[2])
        init_price *= pred[1]
    pred_results.append([max(ups), init_price, min(downs)])


model.summary()  # 파라미터 수 및 깊이 수 확인
print(f"Check\n{y_test[-1][-3:]}")
print(f"Expected Test Data\n{test_results}")
print(f"Predictions\n{pred_results}")
