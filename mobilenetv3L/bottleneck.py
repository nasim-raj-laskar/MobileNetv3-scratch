from tensorflow.keras import layers       #type: ignore
from .conv import ConvBlock
from .se import SEBlock
from .activations import hard_swish

def BNeck(x, in_ch, exp_ch, out_ch, kernel, stride, use_se, activation):

    x_exp = ConvBlock(x, exp_ch, 1, act=activation)

    x_dw = layers.DepthwiseConv2D(
        kernel,
        strides=stride,
        padding='same',
        use_bias=False
    )(x_exp)

    x_dw = layers.BatchNormalization()(x_dw)
    x_dw = layers.ReLU()(x_dw) if activation == 'relu' else layers.Activation(hard_swish)(x_dw)

    if use_se:
        x_dw = SEBlock(x_dw, exp_ch)

    x_proj = layers.Conv2D(out_ch, 1, padding='same', use_bias=False)(x_dw)
    x_proj = layers.BatchNormalization()(x_proj)

    if stride == 1 and in_ch == out_ch:
        x_proj = layers.Add()([x_proj, x])

    return x_proj
