import os.path as op
import tensorflow as tf
from tensorflow.python.ops.gen_parsing_ops import parse_example
from config import Config as cfg
import matplotlib.pyplot as plt

class tfrecord_reader:
    def load_dataset(self, dataname, split):
        tfr_files = tf.io.gfile.glob(op.join(cfg.HARD_ROOT, f"{dataname}_{split}", "*.tfrecord"))
        tfr_files.sort()
        print("[Load Tfrecord files : ", tfr_files)
        dataset = tf.data.TFRecordDataset(tfr_files)
        dataset = dataset.map(parse_example)
        dataset = self.set_data_option(dataset, shuffle=False, batch_size = cfg.Model.batch_size, epochs = 3)
        return dataset

    def parse_example(self, example):
        features = {
            "image" : tf.io.FixedLenFeature(shape=(), dtype=tf.string, default_value=""),
            "label" : tf.io.FixedLenFeature(shape=(), dtype=tf.int64, default_value=0),
            "maskset" : tf.io.FixedLenFeature(shape=(), dtype=tf.int64, default_value=0)
        }

        parsed = tf.io.parse_single_example(example, features)
        #   method 1. decode from string image
        parsed["image"] = tf.io.decode_raw(parsed["image"], tf.uint8)
        #   only viusalize
        parsed["image_uint8"] = tf.reshape(parsed["image"], cfg.Model.img_shape)
        parsed["image"] = tf.io.decode_raw(parsed["image_uint8"], dtype=tf.float32)
        # method 2. decode from png format
        # parsed["image"] = tf.io.decode_png(parsed["image"])
        return parsed

    def set_data_option(self, dataset, shuffle=False, batch_size, epochs):
        if shuffle:
            dataset = dataset.shuffle(100)
        dataset = dataset.batch(batch_size)
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


def tf_main():
    tfr = tfrecord_reader()
    dataset = tfr.load_dataset("cityspace", "train")
    show_samples(dataset)

def show_samples(data):
    for features in data:
        img = features["image_uint8"]
        mask = features["label"]
        plt.imshow(img)
        plt.show()

if __name__ == "__main__":
    main()