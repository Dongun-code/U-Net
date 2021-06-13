import os.path as op
# import tensorflow as tf
from config import Config as cfg
from PIL import Image, ImageOps
import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from sklearn.cluster import KMeans

class data_factory:
    def __init__(self, dataset):
        if dataset == "cityspace":
            return CitySpace()

    
class CitySpace():
    def kmeans(self, img_list):
        img_list.sort()
        mask_list = []
        for name in img_list:
            img = Image.open(name)
            img = np.array(img, dtype=np.float32)
            img = self.cut_img(img)
            mask = img[:,256:, :]
            mask = mask / 255

            mask_list.append(mask.reshape(mask.shape[0]*mask.shape[1], 3))

        mask_list = np.array(mask_list)
        mask = mask_list.reshape((-1, 3))
        kmeans = KMeans(10)
        kmeans.fit(mask)

        return kmeans


        return

    def collect_img(self, img_list):
        img_list.sort()
        climg_list = []
        mask_list = []
        
        for name in img_list:
            img = Image.open(name)
            img = np.array(img, dtype=np.float32)
            img = self.cut_img(img)
            climg = img[:, :256, :]
            mask = img[:,256:, :]

            climg = climg / 255
            mask = mask / 255
            climg_list.append(climg)
            mask_list.append(mask)

        climg_list = np.array(climg_list)
        mask_list = np.array(mask_list)

        return climg_list, mask_list

    def cut_img(self, img):
        img = img[:-56, :, :]
        return img

    def colorCluster(self, img):
        pass


    def __call__(self, path):

        train_path = op.join(cfg.CITY_DATA, "train","*.jpg")
        val_path = op.join(cfg.CITY_DATA, "valid","*.jpg")

        train_data = glob.glob(train_path)
        val_data = glob.glob(val_path)
        train_data.sort()
        # train_data = np.array(train_data)
        # plt.imshow(img)
        # plt.show()
        kmean = self.kmeans(train_data[:30])
        train_img, train_mask = self.collect_img(train_data)
        # val_img, train_mask = self.collect_img(val_data)

        print(train_mask[20].reshape(-1, 3).shape)
        result = kmean.predict(train_mask[50].reshape(-1, 3))
        print(result.shape)
        print(result)
        result = result.reshape((200,256))
        print(result)

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        axes[0].imshow(result)
        axes[1].imshow(train_mask[50])
        plt.show()



if __name__ == "__main__":
    city = CitySpace()
    city(cfg.CITY_DATA)








