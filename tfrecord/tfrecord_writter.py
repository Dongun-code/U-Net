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
        target = np.load(label_set)
        print("Load Data Done")
        img = image['img']
        mask, split_mask = target['label'], target['masket']
        writer = None
        serializer = TfSerializer()
        shard = cfg.SHARD
        
        if not os.path.exists(op.join(cfg.TFRECORD, data_name)):
            print("Make folder")
            os.makedirs(op.join(cfg.TFRECORD,data_name))

        for i, (x, y, mask) in enumerate(zip(img, mask, split_mask)):
            if i % shard == 0:
                writer = self.close_tfr_writer(writer, tfrecord_path, data_name, split, i//shard)
            print(f"[TFRecord write : {i}")
            example = {"image": x, "label": y, "maskset": mask}
            serialized = serializer(example)
            writer.write(serialized)

        writer.close()


    def close_tfr_writer(self, writer, tfrecord_path, data_name, split, index):
        if writer:
            writer.close()

def write_tfrecord(img_path, label_path):
    tfw = tfrecord_writer()
    tfw.make_tfrecord(img_path,label_path, "cityspace", "train", cfg.TFRECORD)


if __name__ == "__main__":
    img_path = op.join(cfg.RESULT_ROOT,"train_img.npz")
    label_path = op.join(cfg.RESULT_ROOT,"train_label.npz")
    write_tfrecord(img_path, label_path)