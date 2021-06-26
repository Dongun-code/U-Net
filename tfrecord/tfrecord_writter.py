import numpy as np
import tensorflow as tf
import sys
sys.path.append('../')
from config import Config as cfg
from tfrecord.setup import TfSerializer
import os
import os.path as op


class tfrecord_writer:
    def __init__(self):
        pass

    def make_tfrecord(self, data_set, label_set, data_name, split, tfrecord_path):
        print("Load Data")
        image = np.load(data_set)
        target = np.load(label_set, encoding="bytes")
        print("Load Data Done")

        img = image["img"]
        print(img.shape)
        print(type(img))
        mask, split_mask = target['label'], target['masket']
        print(mask.shape)
        print(type(mask))
        print(split_mask.shape)
        print(type(split_mask))
        writer = None
        serializer = TfSerializer()
        shard = cfg.SHARD
        
        # if not os.path.exists(op.join(cfg.TFRECORD, data_name)):
        #     print("Make folder")
        #     os.makedirs(op.join(cfg.TFRECORD,data_name))

        for i, (x, y, mask) in enumerate(zip(img, mask, split_mask)):
            # print("shape : ", x.shape, y.shape, mask.shape)
            if i % shard == 0:
                writer = self.close_tfr_writer(writer, tfrecord_path, data_name, split, i//shard)
            print(f"[TFRecord write : {i}")
            example = {"image": x, "label": y, "maskset": mask}
            serialized = serializer(example)
            writer.write(serialized)

        example_proto = tf.train.Example.FromString(serialized)
        # print(example_proto)
        writer.close()


    def close_tfr_writer(self, writer, tfrecord_path, data_name, split, index):
        if writer:
            writer.close()

        tfr_path = op.join(cfg.TFRECORD, data_name)
        tfrdata_path = os.path.join(tfrecord_path, f"{data_name}_{split}")
        if os.path.isdir(tfr_path) and not os.path.isdir(tfrdata_path):
            os.makedirs(tfrdata_path)
        tfrfile = os.path.join(tfrdata_path, f"shard_{index:03d}.tfrecord")
        writer = tf.io.TFRecordWriter(tfrfile)
        print(f"create tfrecord file: {tfrfile}")
        return writer

def write_tfrecord(name, img_path, label_path):
    tfw = tfrecord_writer()
    # tfw.make_tfrecord(img_path,label_path, "cityspace", "train", cfg.HARD_ROOT)
    tfw.make_tfrecord(img_path,label_path, "cityspace", name, cfg.HARD_ROOT)
    # tfw.make_tfrecord(img_path,label_path, "cityspace", "val", cfg.HARD_ROOT)
    # tfw.make_tfrecord(img_path,label_path, "cityspace", "test", cfg.HARD_ROOT)


if __name__ == "__main__":
    data = {
            # 'train' : (op.join(cfg.HARD_ROOT,"train_img.npz"), op.join(cfg.HARD_ROOT,"train_label.npz")),
            'val' : (op.join(cfg.HARD_ROOT,"val_img.npz"), op.join(cfg.HARD_ROOT,"val_label.npz")),
            # 'test' : (op.join(cfg.HARD_ROOT,"test_img.npz"), op.join(cfg.HARD_ROOT,"test_label.npz")),
            }
    for name, file in data.items():
        #  print(file)
        #  print(name)
        write_tfrecord(name, file[0], file[1])