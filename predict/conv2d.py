import numpy as np
import pandas as pd
from keras import callbacks

from models import Conv2DModel
from process import generate_x_data_conv2d, generate_y_data_conv2d
from model_test import plot_result


# 프로젝트 설정
# drive_dir = "drive/My Drive/Colab Notebooks/"
project_name = "conv2d_0"
model_dir = "model/conv2d_0.keras"
data_dir = "data/conv2d.csv"
cluster_dir = "data/clustered_data_fit_7d.csv"

# 변수 설정
test_cnt: int = 120
epochs: int = 100
x_days: int = 7
x_cols: list = [
    "volume_d200",
    "volume_d50",
    "volume_delta",
    "d200",
    "d50",
    "down_delta",
    "delta",
    "up_delta",
]
y_cols: list = ["open", "close", "high", "low"]
activation: str = "relu"
cluster_num: int = 0

# 데이터 가져오기
data = pd.read_csv(data_dir, usecols=x_cols + y_cols, dtype=np.float32)

# 데이터 전처리 및 생성
x_gen = generate_x_data_conv2d(data, x_cols, x_days, 1)
y_gen = generate_y_data_conv2d(data, y_cols, x_days, 1)

x_data = np.array(list(x_gen), dtype=np.float32)
y_data = np.array(list(y_gen), dtype=np.float32)

if len(x_data) != len(y_data):
    raise Exception("Data size mismatch")

# 분류 결과 가져오기
cluster = pd.read_csv(cluster_dir)

# 분류
cluster = cluster[-len(x_data) :]
# 클러스터 번호에 해당하는 인덱스만 추출
indices = cluster.index[cluster["Cluster"] == cluster_num].tolist()
# 해당 인덱스를 사용하여 x_data와 y_data 필터링
x_data = x_data[indices]
y_data = y_data[indices]

# 데이터를 Conv2D 입력에 맞게 4차원으로 변환
x_data = x_data.reshape((x_data.shape[0], x_data.shape[1], x_data.shape[2], 1))
x_shape_input = (x_data.shape[1], x_data.shape[2], 1)
y_shape_input = (None, y_data.shape[1])

# 학습 데이터, 테스트 데이터 분리
x_data_learn = x_data[:-test_cnt]
y_data_learn = y_data[:-test_cnt]
x_data_test = x_data[-test_cnt:]
y_data_test = y_data[-test_cnt:]

print(f"Shape: {x_data_learn.shape},{y_data_learn.shape}")
print(f"Test Shape: {x_data_test.shape},{y_data_test.shape}")

model = Conv2DModel(
    x_shape_input=x_shape_input,
    y_shape_input=y_shape_input,
    activation=activation,
).build()

# 조기 종료 콜백 설정
early_stopping = callbacks.EarlyStopping(
    monitor="val_loss", patience=3, restore_best_weights=True
)

model.fit(
    x_data_learn,
    y_data_learn,
    epochs=epochs,
    callbacks=[early_stopping],
    validation_split=0.2,
)

# 모델 평가(학습 데이터)
result = model.evaluate(x_data_learn, y_data_learn, return_dict=True)
print(f"Results: {result}")
model.summary()  # 파라미터 수 및 깊이 수 확인
model.save(model_dir)

# 예측(테스트 데이터)
pred_results = model.predict(x_data_test)
test_results = y_data_test.copy()

# 예측 결과 출력
plot_result(test_results, pred_results)
test_means = np.mean(test_results, axis=0)
pred_means = np.mean(pred_results, axis=0)
print(f"Test Means:{test_means}")
print(f"Test Last:{test_results[-1]}")
print(f"Pred Means:{pred_means}")
