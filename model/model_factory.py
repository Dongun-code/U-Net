import tensorflow as tf
from tensorflow.python.keras.backend import binary_crossentropy
from tensorflow.python.keras.models import Model
from tensorflow.python.ops.gen_batch_ops import batch
from model.backbone import Unet
from tensorflow.keras import optimizers

class model_factory:
    def __init__(self, backbone, input_shape, batch_size):
        self.backbone = backbone
        self.input_shape = input_shape
        self.batch_size = batch_size
    
    def choose_backbone(self, backbone):
        if backbone == "unet":
            return Unet()
        else:
            raise MyExceptionToCatch(f"[backbone_factory] invalid backbone name: {backbone}")

    def get_model(self):
        model = self.choose_backbone(self.backbone)
        model = model.get_model(self.input_shape, self.batch_size)
        model.compile(optimizer= optimizers.Adam(lr = 1e-4), loss='binary_crossentropy', metrics=['accuracy'])
        model.summary()
        print("out")
        return model


    
