from pprint import pprint
import numpy as np
import tensorflow as tf


class SPostModel(tf.keras.Model):
    def __init__(self, pred_model, resize_shape, *args, **kwargs):
        super(SPostModel, self).__init__(*args, **kwargs)
        self.pred_model = pred_model
        self.resize_shape = tf.cast(resize_shape, tf.float32)

    @tf.function
    def call(self, x, training=False):
        imgs, origin_shapes = x
        batch_size = tf.shape(imgs)[0]
        self.resize_ratio = tf.cast(origin_shapes / self.resize_shape,
                                    tf.dtypes.float32)
        preds = self.pred_model(imgs, training=False)
        b_cls = tf.math.softmax(preds['cls'])
        b_idx = tf.math.argmax(b_cls, axis=-1)
        return b_idx
