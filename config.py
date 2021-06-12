import numpy as np
import os.path as op

class Config:
    RESULT_ROOT = "/home/milab/machine_ws/Semantic Segmentation/Dataset"
    TFRECORD = op.join(RESULT_ROOT, "tfrecord")
    CITY_DATA = op.join(RESULT_ROOT, "cityscapes_data")

    class Tfrecord:
        pass

    class Model:
        color_map = {
            '0': [0, 0, 0],
            '1': [153, 153, 0],
            '2': [255, 204, 204],
            '3': [255, 0, 127],
            '4': [0, 255, 0],
            '5': [0, 204, 204],
            '6': [255, 0, 0],
            '7': [0, 0, 255]
            }
        input_shape = (256, 256, 3)
        model = "unet"
        batch_size = 2
    


