from mobilenetv3L.model import MobileNetV3_Large
import tensorflow as tf

num_classes =                                                                    # specify number of classes for your dataset

model = MobileNetV3_Large(num_classes)
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)
