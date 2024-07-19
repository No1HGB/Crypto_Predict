from keras import callbacks
from keras_tuner import RandomSearch
import keras_tuner as kt

import config
import models
from fetch import fetch_data
from process import cal_value, create_x_data, create_y_data

"""
데이터 개수 늘리기
distance 빼기(create_x_data)
epochs 최대 수를 조절
"""

# 변수 설정
symbol = config.symbol
test_cnt: int = 7
epochs: int = 1000

# 데이터 가져오기 및 처리
data = fetch_data(symbol=symbol, interval="5m", numbers=37000)
data = cal_value(data)
data_future = fetch_data(symbol=symbol, interval="5m", numbers=37000, type="future")
data_future = cal_value(data_future)

x_data = create_x_data(data, data_future)
y_data = create_y_data(data, data_future)

x_shape_input: tuple = (x_data.shape[1], x_data.shape[2])
y_shape_input: int = y_data.shape[1]

x_data_learn = x_data[:-test_cnt]
y_data_learn = y_data[:-test_cnt]
x_data_test = x_data[-test_cnt:]
y_data_test = y_data[-test_cnt:]

# RandomSearch 객체 생성
hypermodel = models.RegressionHyperModel(
    x_shape_input=x_shape_input, y_shape_input=y_shape_input
)
tuner = RandomSearch(
    hypermodel,
    objective="val_loss",
    max_trials=10,
    executions_per_trial=3,  # 하이퍼파라미터 조합에 대해 모델 실행 횟수(3번이면, 3번 실행 후 평균)
    directory="hyperparam",
    project_name="sequential_random",
)

# 조기 종료 콜백 설정
early_stopping = callbacks.EarlyStopping(
    monitor="val_loss", patience=16, restore_best_weights=True
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

# 최적의 하이퍼파라미터 출력
best_hps = tuner.get_best_hyperparameters(num_trials=1)[
    0
]  # 최적 하이퍼파라미터 개수, 첫 번째
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
print(f"Shape: {x_data.shape},{y_data.shape}")
print(f"Test Shape: {x_data_test.shape},{y_data_test.shape}")
print(f"Results: {result}")
model.summary()  # 파라미터 수 및 깊이 수 확인

# 예측(테스트 데이터)
y_pred = model.predict(x_data_test)

# 예측 결과 출력
print(f"Expected Test Data\n{y_data_test}")
print(f"Predictions\n{y_pred}")
