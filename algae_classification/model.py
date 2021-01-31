from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2


class AlgaeClassificationModel:

    def __init__(self):
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3),
        )

        inputs = Input(shape=(224, 224, 3))

        x = base_model(inputs, trainable=False)
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.2)(x)
        output = Dense(2, activation='softmax')(x)

        self.model = Model(input, output)
        self.compile()

    def compile(self, learning_rate=0.001):
        self.model.compile(
            optimizer=Adam(learning_rate),
            loss=CategoricalCrossentropy(from_logits=True),
            metrics=[CategoricalAccuracy()]
        )

    def unfreeze(self, learning_rate):
        self.model.trainable = True
        self.compile(learning_rate)

    def freeze(self, learning_rate):
        self.model.trainable = False
        self.compile(learning_rate)
