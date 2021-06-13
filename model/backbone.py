import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.keras.models import Model
from model.model_util import CustomConv2d, CustomConv2dup


def backbone_factory(backbone):
    if backbone == "Unet":
        return Unet()
    else:
        raise MyExceptionToCatch(f"[backbone factory] invalid backbone name: {backbone}")

class Backbone:
    def __init__(self):
        self.conv2d = CustomConv2d(kernel_size=3, strides=1)
        self.conv2dup = CustomConv2dup(kernel_size=2, strides=1)

    def Maxpooling(self, x):
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        return x

    def Dropout(self, x, ratio):
        x = layers.Dropout(ratio)(x)
        return x

    def conv2d_k1(self, x):
            x = layers.Conv2D(1, 1, activation="softmax")(x)
            return x


class Unet(Backbone):
    def __init__(self):
        super().__init__()
    
    def get_model(self, input_shape, batch_size):        
        input = tf.keras.Input(shape=input_shape, batch_size=batch_size)
        
        down0 = self.conv2d(input, 64)
        down1 = self.conv2d(down0, 64)
        max0 = self.Maxpooling(down1)

        down2 = self.conv2d(max0, 128)
        down3 = self.conv2d(down2, 128)
        max1 = self.Maxpooling(down3)

        down4 = self.conv2d(max1, 256)
        down5 = self.conv2d(down4, 256)
        max2 = self.Maxpooling(down5)

        down6 = self.conv2d(max2, 512)
        down7 = self.conv2d(down6, 512)
        down7 = self.Dropout(down7, 0.5)
        max3 = self.Maxpooling(down7)

        down8 = self.conv2d(max3, 1024)
        down9 = self.conv2d(down8, 1024)
        down9 = self.Dropout(down9, 0.5)

        up0 = self.conv2dup(down9, 512)
        merge0 = layers.concatenate([down7, up0], axis=3)
        up1 = self.conv2d(merge0, 512)
        up2 = self.conv2d(up1, 512)

        up3 = self.conv2dup(up2, 256)
        merge1 = layers.concatenate([down5, up3])
        up4 = self.conv2d(merge1, 256)
        up5 = self.conv2d(up4, 256)

        up6 = self.conv2dup(up5, 128)
        merge2 = layers.concatenate([down3, up6])
        up7 = self.conv2d(merge2, 128)
        up8 = self.conv2d(up7, 128)

        up9 = self.conv2dup(up8, 64)
        merge3 = layers.concatenate([down1, up9])
        up10 = self.conv2d(merge3, 13)
        output = layers.Conv2D(13, 1, activation="softmax")(up10)

        model = tf.keras.Model(inputs = input, outputs=output)
        return model



        #   BottleNeck
