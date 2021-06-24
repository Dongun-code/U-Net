import tensorflow as tf
import numpy as np


class TfSerializer:
    def __call__(self, raw_example):
        features = self.convert_to_features(raw_example)

        features = tf.train.Features(feature=features)

        tf_example = tf.train.Example(features = features)

        serialized = tf_example.SerializeToString()

        return serialized

    def convert_to_features(self, raw_example):
        features = dict()
        for key, value in raw_example.items():
            if value is None:
                continue
            elif isinstance(value, np.ndarray):
                # method 1: encode into raw bytes - fast but losing shape, 2 seconds to make training dataset
                value = value.tobytes()
                # method 2: encode into png format - slow but keeping shape, 10 seconds to make training dataset
                # value = tf.io.encode_png(value)
                # value = value.numpy()  # BytesList won't unpack a tf.string from an EagerTensor.
                features[key] = self._bytes_feature(value)
            elif isinstance(value, str):
                value = bytes(value, 'utf-8')
                features[key] = self._bytes_feature(value)
            elif isinstance(value, int):
                features[key] = self._int64_feature(value)
            elif isinstance(value, float):
                features[key] = self._float_feature(value)
            else:
                assert 0, f"[convert_to_feature] Wrong data type: {type(value)}"
        return features

    @staticmethod
    def _bytes_feature(value):
        """Returns a bytes_list from a string / byte."""
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @classmethod
    def _float_feature(value):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    @classmethod
    def _int64_feature(value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))