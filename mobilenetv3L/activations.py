import tensorflow as tf

def hard_swish(x):
    return x * tf.nn.relu6(x + 3) / 6
