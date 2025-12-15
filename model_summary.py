import tensorflow as tf
from mobilenetv3L.model import MobileNetV3_Large

if __name__ == "__main__":
    model = MobileNetV3_Large(
        input_shape=(224, 224, 3),
        num_classes=1000
    )
    model.summary()
