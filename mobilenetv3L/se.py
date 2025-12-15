from tensorflow.keras import layers   # type: ignore

def SEBlock(x, filters, reduction=4):
    """
    Squeeze-and-Excitation block
    """
    # Squeeze
    se = layers.GlobalAveragePooling2D()(x)

    # Excitation
    se = layers.Dense(filters // reduction, activation='relu')(se)
    se = layers.Dense(filters, activation='hard_sigmoid')(se)

    # Scale
    se = layers.Reshape((1, 1, filters))(se)
    return layers.Multiply()([x, se])
