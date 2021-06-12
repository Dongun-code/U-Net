import os.path as op

from config import Config as cfg
from PIL import Image
import glob
import numpy as np
import matplotlib.pyplot as plt

class data_factory:
    def __init__(self, dataset):
        if dataset == "cityspace":
            return CitySpace()

    
class CitySpace():
    def __call__(self, path):
        train_path = op.join(cfg.CITY_DATA, "train","*.jpg")
        val_path = op.join(cfg.CITY_DATA, "valid","*.jpg")

        train_data = glob.glob(train_path)
        val_data = glob.glob(val_path)

        img = Image.open(train_data[0])
        img = np.array(img)

        climg = img[:, :256, :]
        mask = img[:,256:, :]

        print(climg.shape)
        print(mask)
        plt.imshow(mask)
        plt.show()



if __name__ == "__main__":
    city = CitySpace()
    city(cfg.CITY_DATA)








