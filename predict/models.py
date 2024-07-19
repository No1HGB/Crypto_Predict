from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Input
from keras_tuner import HyperModel


class RegressionHyperModel(HyperModel):
    def __init__(self, x_shape_input: tuple, y_shape_input: int):
        super().__init__()  # 부모 클래스의 __init__ 메서드 호출
        self.x_shape_input = x_shape_input
        self.y_shape_input = y_shape_input

    def build(self, hp):
        # 모델 구축
        model = keras.Sequential()
        # 입력층
        model.add(Input(shape=self.x_shape_input))

        # 은닉층 수를 조절
        for i in range(hp.Int("num_layers", 1, 5)):
            model.add(
                Dense(
                    units=hp.Int(
                        "units_" + str(i), min_value=32, max_value=512, step=32
                    ),
                    activation="relu",
                )
            )
        # 출력층 (회귀식이므로 출력층 활성함수 없음)
        model.add(Dense(self.y_shape_input))
        # 모델 컴파일
        # 학습률 조절
        learning_rate = hp.Float(
            "learning_rate", min_value=1e-4, max_value=1e-2, sampling="log"
        )

        # 모델 컴파일
        model.compile(
            loss="mean_squared_error", optimizer=Adam(learning_rate=learning_rate)
        )
        return model
