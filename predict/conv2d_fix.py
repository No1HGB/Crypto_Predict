import numpy as np
import pandas as pd
from keras import callbacks

import models
from process import create_x_data_conv2d, create_y_data_conv2d


# 프로젝트 설정
project_name = "conv2d_fix"
model_dir = "model/" + "conv2d_fix.keras"

# 데이터 가져오기
data = pd.read_csv("data/conv2d.csv")

# 변수 설정
data_cnt: int = len(data)
test_cnt: int = 12
epochs: int = 100
x_days: int = 3
x_cols: list = ["volume_ratio", "down_delta", "delta", "up_delta"]
y_cols: list = ["down_delta", "delta", "up_delta"]
activation: str = "leaky_relu"
important_cnt = 12000
regular_weight = 1.0
important_weight = regular_weight * len(data) / (important_cnt * 2)

data = data[-data_cnt:]

# 데이터 전처리 및 생성
x_data = create_x_data_conv2d(data, x_cols, x_days, 1)
y_data = create_y_data_conv2d(data, y_cols, x_days, 1)

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

print(f"Shape: {x_data_learn.shape},{y_data_learn.shape}")
print(f"Test Shape: {x_data_test.shape},{y_data_test.shape}")

model = models.Conv2DModel(
    x_shape_input=x_shape_input,
    y_shape_input=y_shape_input,
    activation=activation,
).build()

# 조기 종료 콜백 설정
early_stopping = callbacks.EarlyStopping(
    monitor="val_loss", patience=3, restore_best_weights=True
)

# 가중치 배열
sample_weight = np.ones(data_cnt)
sample_weight[-important_cnt:] = important_weight
sample_weight = sample_weight.reshape(-1, 1, 1, 1)
sample_weight = np.tile(sample_weight, (1, y_data.shape[1], y_data.shape[2], 1))

model.fit(
    x_data_learn,
    y_data_learn,
    epochs=epochs,
    callbacks=[early_stopping],
    validation_split=0.2,
    sample_weight=sample_weight,
)

# 모델 평가(학습 데이터)
result = model.evaluate(x_data_learn, y_data_learn, return_dict=True)
print(f"Results: {result}")
model.summary()  # 파라미터 수 및 깊이 수 확인
model.save(model_dir)


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

print(f"Expected Test Data\n{test_results}")
print(f"Predictions\n{pred_results}")
