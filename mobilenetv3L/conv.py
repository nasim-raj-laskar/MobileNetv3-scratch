from tensorflow.keras import layers  # type: ignore
from .activations import hard_swish

def ConvBlock(x, filters, kernel, stride=1, act='hardswish'):
    x = layers.Conv2D(
        filters,
        kernel,
        strides=stride,
        padding='same',
        use_bias=False,
        kernel_initializer='he_normal'
    )(x)
    x = layers.BatchNormalization()(x)

    if act == 'relu':
        x = layers.ReLU()(x)
    else:
        x = layers.Activation(hard_swish)(x)

    return x