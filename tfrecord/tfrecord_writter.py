import numpy as np
import tensorflow as tf
from config import Config as cfg
from tfrecord.setup import TfSerializer
import os

class tfrecord_writer:
    def __init__(self):
        pass

    def make_tfrecord(self, data_set, data_name, split, tfrecord_path):
        img, maskset = data_set
        label = maskset[0]
        mask_ = maskset[1]
        writer = None
        serializer = TfSerializer()
        shard = cfg.SHARD

        if not os.path.exists(cfg.TFRECORD):
            os.makedirs(cfg.TFRECORD)

        for i, (x, y, mask) in enumerate(zip(img, label, mask_)):
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

# def write_tfrecord():
#     tfw = tfrecord_writter()
#     train_set, test_set = get_data()
#     tfw.make_tfrecord(train_set, "cityspace", "train", cfg.TFRECORD)
