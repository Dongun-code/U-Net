import os.path as op
import tensorflow as tf
import sys

from tensorflow.python.ops.gen_math_ops import imag
sys.path.append('../')
# from tensorflow.python.ops.gen_parsing_ops import parse_example
from config import Config as cfg
import matplotlib.pyplot as plt


class tfrecord_reader:
    def load_dataset(self, dataname, split):
        tfr_files = tf.io.gfile.glob(op.join(cfg.HARD_ROOT, f"{dataname}_{split}", "*.tfrecord"))
        tfr_files.sort()
        print("[Load Tfrecord files : ", tfr_files)
        dataset = tf.data.TFRecordDataset(tfr_files)
        dataset = dataset.map(self.parse_example)
        dataset = self.set_data_option(dataset, shuffle=False, batch_size = cfg.Model.batch_size, epochs = 3)
        return dataset

    def parse_example(self, example):
        features = {
            "image" : tf.io.FixedLenFeature(shape=(), dtype=tf.string),
            "label" : tf.io.FixedLenFeature(shape=(), dtype=tf.string),
            "maskset" : tf.io.FixedLenFeature(shape=(), dtype=tf.string)
        }

        parsed = tf.io.parse_single_example(example, features)
        print("Data : ", parsed["image"])
        parsed["image"] = tf.io.decode_raw(parsed["image"], tf.float64)
        parsed["image_u8"] = tf.reshape(parsed["image"], [256, 512, 3])

        parsed["label"] = tf.io.decode_raw(parsed["label"], tf.uint8)
        parsed["label"] = tf.reshape(parsed["label"], [256, 512])

        parsed["maskset"] = tf.io.decode_raw(parsed["maskset"], tf.float64)
        parsed["maskset"] = tf.reshape(parsed["maskset"], [256, 512, 3])

        # #   only viusalize
        # method 2. decode from png format
        # parsed["image"] = tf.io.decode_png(parsed["image"])
        return parsed

    def set_data_option(self, dataset, shuffle, batch_size, epochs):
        if shuffle:
            dataset = dataset.shuffle(100)
        dataset = dataset.batch(1)
        return dataset

def gpu_config():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must 
            print(e)

def show_samples(data):
    for features in data:
        # img = features["image_uint8"].numpy()
        # print("shpae : ", img.shape)
        # print("Shape :", img.numpy())
        img = features["image_u8"][0]
        mask = features["label"]
        plt.imshow(img)
        plt.show()

def tf_main():
    tfr = tfrecord_reader()
    dataset = tfr.load_dataset("cityspace", "val")
    show_samples(dataset)



if __name__ == "__main__":
    tf_main()