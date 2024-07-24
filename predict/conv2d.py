import pandas as pd
from keras import callbacks
from keras_tuner import RandomSearch
import keras_tuner as kt
import matplotlib.pyplot as plt

import models
from process import create_x_data_conv2d, create_y_data_conv2d, inverse_scaling


# 프로젝트 설정
project_name = "conv2d_1"
model_dir = "model/" + "conv2d_1.keras"

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
x_data = create_x_data_conv2d(data, x_cols, 3)
y_data, min_max_values_list = create_y_data_conv2d(data, y_cols, 3, 1)

# 데이터를 Conv2D 입력에 맞게 4차원으로 변환
x_data = x_data.reshape((x_data.shape[0], x_data.shape[1], x_data.shape[2], 1))
y_data = y_data.reshape((y_data.shape[0], y_data.shape[1], y_data.shape[2], 1))
x_shape_input = (x_data.shape[1], x_data.shape[2], 1)
y_shape_input = (y_data.shape[1], y_data.shape[2], 1)

# 학습 데이터, 테스트 데이터 분리
x_data_learn = x_data[:-test_cnt]
y_data_learn = y_data[:-test_cnt]
min_max_values_list_learn = min_max_values_list[:-test_cnt]

x_data_test = x_data[-test_cnt:]
y_data_test = y_data[-test_cnt:]
min_max_values_list_test = min_max_values_list[-test_cnt:]

print(f"Shape: {x_data_learn.shape},{y_data_learn.shape}")
print(f"Test Shape: {x_data_test.shape},{y_data_test.shape}")


# RandomSearch 객체 생성
hypermodel = models.Conv2DHyperModel(
    x_shape_input=x_shape_input,
    y_shape_input=y_shape_input,
    name=project_name,
    activation=activation,
)

# 하이퍼파라미터 서치 객체(튜너) 생성
tuner = RandomSearch(
    hypermodel,
    objective="val_loss",
    max_trials=10,
    executions_per_trial=1,  # 하이퍼파라미터 조합에 대해 모델 실행 횟수(3번이면, 3번 실행 후 평균)
    directory="hyperparam",
    project_name=project_name,
)

# 조기 종료 콜백 설정
early_stopping = callbacks.EarlyStopping(
    monitor="val_loss", patience=3, restore_best_weights=True
)  # 학습 시 validation_split에 따라 val_loss, loss 선택

# 하이퍼파라미터 튜닝 수행
hp = kt.HyperParameters()
tuner.search(
    x_data_learn,
    y_data_learn,
    epochs=epochs,
    validation_split=0.2,
    callbacks=[early_stopping],
    batch_size=hp.Int("batch_size", 16, 256, step=16),
)

# 최적의 하이퍼파라미터 출력(최적 하이퍼파라미터 개수, 첫 번째)
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"Best number of layers: {best_hps.get('num_layers')}")
for i in range(best_hps.get("num_layers")):
    print(f"Layer {i} units: {best_hps.get('units_' + str(i))}")

# 최적의 하이퍼파라미터로 모델 학습
model = tuner.hypermodel.build(best_hps)
history = model.fit(
    x_data_learn,
    y_data_learn,
    epochs=epochs,
    callbacks=[early_stopping],
    validation_split=0.2,
)  # validation_split 적용 여부 고려

# 모델 평가(학습 데이터)
result = model.evaluate(x_data_learn, y_data_learn, return_dict=True)
print(f"Results: {result}")
model.summary()  # 파라미터 수 및 깊이 수 확인
model.save(model_dir)

# 예측(테스트 데이터)
y_result = model.predict(x_data_test)
y_pred = inverse_scaling(y_result, min_max_values_list_test)
y_pred_origin = inverse_scaling(
    y_result, min_max_values_list_learn[-test_cnt:]
)  # 실전에서는 예측 데이터의 minmaxlist는 존재하지 않으므로 input 데이터의 list 사용

# 예측 결과 출력
x = range(y_data_test.shape[1])
test_vector = y_data_test[-1]
pred_vector = y_pred_origin[-1]
plt.figure(figsize=(10, 6))
plt.plot(x, test_vector, label="Test", color="blue")
plt.plot(x, pred_vector, label="Pred", color="red")
plt.title("Graphs of Two Vectors")
plt.xlabel("Index")
plt.ylabel("Value")
plt.legend()
plt.show()

print(f"Expected Test Data\n{y_data_test[-1]}")
print(f"Predictions\n{y_pred[-1]}")
print(f"Origin Predictions\n{y_pred_origin[-1]}")
