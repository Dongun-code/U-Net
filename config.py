import numpy as np
import os.path as op

class Config:
    RESULT_ROOT = "/home/milab/machine_ws/Semantic Segmentation/Dataset"
    TFRECORD = op.join(RESULT_ROOT, "tfrecord")
    CITY_DATA = op.join(RESULT_ROOT,"leftImg8bit")
    CITY_DATA_LABEL = op.join(RESULT_ROOT,"gt","gtFine")
    USE_CATEGORY = ['car', 'road', 'person']
    COLOR = {'car' : 15,
            'road' : 25,
            'traffic sign' : 35,
            'sidewalk' : 45,
            'terrain' : 55 ,
            'person' : 65}

    class Tfrecord:
        pass

    class CitySpace:
        resolution = (1024, 2048)

    class Model:
        input_shape = (256, 512)
        model = "unet"
        batch_size = 2

    
    


