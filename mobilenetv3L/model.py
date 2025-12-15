from tensorflow.keras import layers, models     #type: ignore
from .conv import ConvBlock 
from .bottleneck import BNeck

def MobileNetV3_Large(input_shape=(224, 224, 3), num_classes=1000):
    inputs = layers.Input(shape=input_shape)
    x = ConvBlock(inputs, 16, 3, stride=2, act='hardswish')

    # Initial Convolution Layer, Input: 224×224×3 → Output: 112×112×16
    x = ConvBlock(inputs, 16, 3, stride=2, act='hardswish')

    # Stage 1 , Shallow feature learning with small filters (3×3)

    x = BNeck(x, 16, 16, 16, 3, 1, False, 'relu')       # stride=1, no SE
    x = BNeck(x, 16, 64, 24, 3, 2, False, 'relu')       # stride=2 → downsample
    x = BNeck(x, 24, 72, 24, 3, 1, False, 'relu')       # stride=1

    # Stage 2 Medium feature extraction; introduce SE and 5×5 depthwise convs

    x = BNeck(x, 24, 72, 40, 5, 2, True, 'relu')        # stride=2, SE on
    x = BNeck(x, 40, 120, 40, 5, 1, True, 'relu')
    x = BNeck(x, 40, 120, 40, 5, 1, True, 'relu')

    # Stage 3 , Deeper representation; switch to Hard-Swish activations

    x = BNeck(x, 40, 240, 80, 3, 2, False, 'hardswish') # stride=2 → downsample
    x = BNeck(x, 80, 200, 80, 3, 1, False, 'hardswish')
    x = BNeck(x, 80, 184, 80, 3, 1, False, 'hardswish')
    x = BNeck(x, 80, 184, 80, 3, 1, False, 'hardswish')
    x = BNeck(x, 80, 480, 112, 3, 1, True, 'hardswish') # SE enabled
    x = BNeck(x, 112, 672, 112, 3, 1, True, 'hardswish')

    # Stage 4, High-level features; larger receptive field (5×5)

    x = BNeck(x, 112, 672, 160, 5, 2, True, 'hardswish') # stride=2 → downsample
    x = BNeck(x, 160, 960, 160, 5, 1, True, 'hardswish')
    x = BNeck(x, 160, 960, 160, 5, 1, True, 'hardswish')

    # Final Layers
    x = ConvBlock(x, 960, 1, act='hardswish')            # 1×1 conv for fusion
    x = layers.GlobalAveragePooling2D()(x)               # Global feature aggregation
    x = layers.Reshape((1, 1, 960))(x)                   # Shape → (1, 1, 960)
    x = ConvBlock(x, 1280, 1, act='hardswish')           # Final 1×1 conv
    x = layers.Dropout(0.8)(x)                           # Regularization
    x = layers.Conv2D(num_classes, 1)(x)                 # Class logits
    x = layers.Flatten()(x)

    return models.Model(inputs, x, name="MobileNetV3_Large")
