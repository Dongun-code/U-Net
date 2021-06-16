import numpy as np
import tensorflow as tf
from config import Config as cfg
from tfrecord.setup import TfSerializer
class tfrecord_writer:
    def __init__(self):
        pass

    def make_tfrecord(self, data_set, data_name, split, tfrecord_path):
        img, label = data_set[:2]
        mask_ = data_set[2:]
        writer = None
        serializer = TfSerializer()
        shard = cfg.SHARD

        for i, (x, y, mask) in enumerate(zip(img, label, mask_)):
            if i % shard == 0:
                writer = self.close_tfr_writer(writer, tfrecord_path, data_name, split, i//shard)

            example = {"image": x, "split_mask": y, "mask": mask}
            serialized = serializer(example)
            writer.write(serialized)

        writer.close()


    def close_tfr_writer(self, writer, tfrecord_path, data_name, split, index):
        if writer:
            writer.close()

def write_tfrecord():
    tfw = tfrecord_writter()
    train_set, test_set = get_data()
    tfw.make_tfrecord(train_set, "cityspace", "train", cfg.TFRECORD)
