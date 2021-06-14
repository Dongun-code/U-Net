import os.path as op
# import tensorflow as tf
from config import Config as cfg
from PIL import Image, ImageOps
import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
# from sklearn.cluster import KMeans

class data_factory:
    def __init__(self, dataset):
        if dataset == "cityspace":
            return CitySpace()

    
class CitySpace():
    def __init__(self):
        self.data_path = cfg.CITY_DATA
        self.label_path = cfg.CITY_DATA_LABEL
        self.obj_color = cfg.COLOR
        self.input_shape = cfg.Model.input_shape

    def img_process(self, img):
        pass
    
    def label_process(self, label):
        pass

    def get_data(self, path_, split):
        path = op.join(path_, split, "*")
        folder = glob.glob(path)
        folder.sort()
        file_list = []
        for fold in folder:
            if "leftImg8bit" in path_:
                path = op.join(fold,"*.png")
            elif "gtFine" in path_:
                path = op.join(fold, "*.json")
            data_list = glob.glob(path)
            data_list.sort()
            file_list.extend(data_list)

        print("file shape : ", np.array(file_list).shape)
        if "leftImg8bit" in path_:
            self.img_process(file_list)
        elif "gtFine" in path_:
            self.label_process(file_list)


        return file_list
        

    def __call__(self, path):

        train = self.get_data(self.data_path, "train")







if __name__ == "__main__":
    city = CitySpace()
    city(cfg.CITY_DATA)








