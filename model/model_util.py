import tensorflow as tf
import tensorflow.keras
from tensorflow.python.keras import layers

class CustomConv2d:
    CALL_COUNT = -1

    def __init__(self, kernel_size = 3, strides=1, padding="same", activation="relu", bn=True):
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.bn = bn


    def __call__(self, x, filters, name = None):
        CustomConv2d.CALL_COUNT += 1
        index = CustomConv2d.CALL_COUNT
        name = f"conv{index:03d}" if name is None else f"{name}/{index:03d}"

        x = layers.Conv2D(filters, self.kernel_size, self.strides, self.padding, use_bias=not self.bn, kernel_initializer = "he_normal", name=name)(x)

        if self.activation == "relu":
            x = layers.ReLU()(x)
        
        if self.bn:
            x = layers.BatchNormalization()(x)
        
        return x

class CustomConv2dup:
    CALL_COUNT = -1

    def __init__(self, kernel_size = 2, strides=1, padding="same", activation="relu", bn=True):
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.bn = bn


    def __call__(self, x, filters, name = None):
        CustomConv2d.CALL_COUNT += 1
        index = CustomConv2d.CALL_COUNT
        name = f"conv_upsample{index:03d}" if name is None else f"{name}/{index:03d}"

        x = layers.Conv2D(filters, self.kernel_size, self.strides, self.padding, use_bias=not self.bn,
                                kernel_initializer = "he_normal", name=name)(layers.UpSampling2D(size = (2, 2))(x))

        if self.activation == "relu":
            x = layers.ReLU()(x)
        
        if self.bn:
            x = layers.BatchNormalization()(x)
        
        return x

