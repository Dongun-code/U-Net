import numpy as np
import os.path as op

class Config:
    RESULT_ROOT = "/home/milab/machine_ws/Semantic Segmentation/Dataset"
    TFRECORD = op.join(RESULT_ROOT, "tfrecord")
    CITY_DATA = op.join(RESULT_ROOT, "cityscapes_data")

    class Tfrecord:
        pass

    class Model:
        colors = [(255,0,0), (0,255,0), (0,0,255),
                (255,255,0), (255,0,255), (0,255,255),
                (255,255,255), (200,50,0),(50,200,0),
                (50,0,200), (200,200,50), (0,50,200),
                (0,200,50), (0,0,0)]
        input_shape = (256, 256, 3)
        model = "unet"
        batch_size = 2
    


