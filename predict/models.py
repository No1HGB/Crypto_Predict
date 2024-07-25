from tensorflow import keras
from tensorflow.keras.layers import (
    Dense,
    Input,
    Conv1D,
    GlobalAveragePooling1D,
    Conv2D,
    MaxPooling2D,
)
from keras_tuner import HyperModel


class RegressionHyperModel(HyperModel):
    def __init__(self, x_shape_input: tuple, y_shape_input: int, name=None):
        super().__init__(name=name)  # 부모 클래스의 __init__ 메서드 호출
        self.x_shape_input = x_shape_input
        self.y_shape_input = y_shape_input

    def build(self, hp):
        # 모델 구축
        model = keras.Sequential(name=self.name)
        # 입력층
        model.add(Input(shape=self.x_shape_input))

        # Conv1D 레이어 추가
        filters = hp.Int("conv1d_filters", min_value=32, max_value=128, step=16)
        kernel_size = hp.Int("conv1d_kernel_size", min_value=2, max_value=5)
        model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation="relu"))
        model.add(GlobalAveragePooling1D())

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
        model.compile(loss="mean_squared_error", optimizer="adam")
        return model


class Conv2DHyperModel(HyperModel):
    def __init__(
        self,
        x_shape_input: tuple,
        y_shape_input: tuple,
        name: str = "conv2d",
        activation: str = "leaky_relu",
    ):
        super().__init__(name=name)  # 부모 클래스의 __init__ 메서드 호출
        self.x_shape_input = x_shape_input
        self.y_shape_input = y_shape_input
        self.activation = activation

    def build(self, hp):
        # 입력층
        inputs = Input(shape=self.x_shape_input)

        # 첫 번째 Conv2D 층
        x = Conv2D(
            filters=hp.Int("conv1_filters", 32, 128, step=32),
            kernel_size=hp.Choice("conv1_kernel", values=[3, 5]),
            activation=self.activation,
            padding="same",
        )(inputs)

        # x shape을 y shape 에 맞게 행 줄이기
        pool_y: int = int(self.x_shape_input[0] / self.y_shape_input[0])
        x = MaxPooling2D(pool_size=(pool_y, 1))(x)

        # 추가 Conv2D 층
        for i in range(hp.Int("num_conv_layers", 1, 3)):
            x = Conv2D(
                filters=hp.Int(f"conv{i + 2}_filters", 32, 128, step=32),
                kernel_size=hp.Choice(f"conv{i + 2}_kernel", values=[3, 5]),
                activation=self.activation,
                padding="same",
            )(x)

        # 출력 층
        out_y: int = self.x_shape_input[1] - self.y_shape_input[1] + 1
        outputs = Conv2D(
            filters=1, kernel_size=(1, out_y), activation="linear", padding="valid"
        )(x)

        model = keras.Model(inputs=inputs, outputs=outputs)

        model.compile(
            optimizer="adam",
            loss="mean_squared_error",
        )

        return model


class Conv2DModel:
    def __init__(
        self,
        x_shape_input: tuple,
        y_shape_input: tuple,
        activation: str = "leaky_relu",
    ):
        super().__init__()  # 부모 클래스의 __init__ 메서드 호출
        self.x_shape_input = x_shape_input
        self.y_shape_input = y_shape_input
        self.activation = activation

    def build(self):
        # 입력층
        inputs = Input(shape=self.x_shape_input)

        # 첫 번째 Conv2D 층
        x = Conv2D(
            filters=96, kernel_size=3, activation=self.activation, padding="same"
        )(inputs)

        # x shape을 y shape 에 맞게 행 줄이기
        pool_y: int = int(self.x_shape_input[0] / self.y_shape_input[0])
        x = MaxPooling2D(pool_size=(pool_y, 1))(x)

        # 추가 Conv2D 층
        x = Conv2D(
            filters=128, kernel_size=5, activation=self.activation, padding="same"
        )(x)

        # 출력 층
        out_y: int = self.x_shape_input[1] - self.y_shape_input[1] + 1
        outputs = Conv2D(
            filters=1, kernel_size=(1, out_y), activation="linear", padding="valid"
        )(x)
        model = keras.Model(inputs=inputs, outputs=outputs)

        model.compile(optimizer="adam", loss="mean_squared_error")

        return model
