import os.path as op
# import tensorflow as tf
from config import Config as cfg
from PIL import Image, ImageOps, ImageDraw
from tfrecord.tfrecord_writter import tfrecord_writer
import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import json

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

    def img_process(self, file_list):
        output = []
        print("Start get img")
        for file in file_list:
            img = Image.open(file)
            img = img.resize((cfg.Model.input_shape[1], cfg.Model.input_shape[0]))
            img = np.array(img)
            output.append(img / 255)
        return output

    def label_process(self, file_list, data_name):
        print("Start get mask")
        mask_list = []
        split_mask_list = []
        if data_name == "cityspace":
            resolution = cfg.CitySpace.resolution
        else:
            raise NameError(f"[Dataset] invalid dataset name: {data_name}")
            
        for file in file_list:
            use_index = set()

            with open(file) as json_file:
                label = json.load(json_file)
            labels = label['objects']
            mask = Image.new('L', (cfg.Model.input_shape[1], cfg.Model.input_shape[0]), 0)
            for label in labels:
                name = label['label']
                if name in cfg.USE_CATEGORY:
                    poly = []
                    polygon = label['polygon']
                    for ob in polygon:
                        poly.append(((ob[0] / resolution[0]) * cfg.Model.input_shape[0] , (ob[1] / resolution[1]) * cfg.Model.input_shape[1]))
                    use_index.add(cfg.COLOR[name])
                    ImageDraw.Draw(mask).polygon(poly, outline=cfg.COLOR[name], fill=cfg.COLOR[name])
            split_mask = self.split_mask(mask, use_index)
            split_mask = np.array(split_mask)
            mask = np.array(mask)
            split_mask_list.append(split_mask)
            mask_list.append(mask)
        return mask_list, split_mask_list
            # plt.imshow(mask)
            # plt.show()

    def split_mask(self, mask, use_index):
        '''
            must think class order
        '''
        split_mask = np.zeros((cfg.Model.input_shape[0], cfg.Model.input_shape[1], len(cfg.USE_CATEGORY)))
        mask = np.array(mask)
        # print('origin : ', split_mask.shape)
        for index in use_index:
            color = (mask == index)
            if index == 15:
                split_mask[:, :, 0] = np.where(color, 15, 0)
            if index == 25:
                split_mask[:, :, 1] = np.where(color, 25, 0)
            if index == 65:
                split_mask[:, :, 2] = np.where(color, 65, 0)
        return split_mask

    def get_data(self, path_, split, data_name):
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
            output = self.img_process(file_list)
        elif "gtFine" in path_:
            output = self.label_process(file_list, data_name)
        return output

    def __call__(self):

        train_img = self.get_data(self.data_path, "train", "cityspace")
        # print(np.array(train_img).shape)
        mask, split_mask = train_label = self.get_data(self.label_path, "train", "cityspace")
        print(np.array(mask).shape)
        print(np.array(split_mask).shape)
        # print(np.array(train_label).shape)
        # val_dataset = self.get_data(self.label_path, "val", "cityspace")
        # test_dataset = self.get_data(self.label_path, "test", "cityspace")
        np.savez("./train_img", img = train_img)
        np.savez("./train_label", label = mask, masket=split_mask)





if __name__ == "__main__":
    city = CitySpace()
    city()








