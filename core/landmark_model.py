from pprint import pprint
import numpy as np
import tensorflow as tf


class LPostModel(tf.keras.Model):
    def __init__(self, pred_model, n_landmarks, resize_shape, *args, **kwargs):
        super(LPostModel, self).__init__(*args, **kwargs)
        self.pred_model = pred_model
        self.n_landmarks = n_landmarks

        self.resize_shape = tf.cast(resize_shape, tf.float32)

    # @tf.function
    def call(self, x, training=False):
        imgs, origin_shapes = x
        batch_size = tf.shape(imgs)[0]
        self.resize_ratio = tf.cast(origin_shapes / self.resize_shape,
                                    tf.dtypes.float32)
        preds = self.pred_model(imgs, training=False)
        b_landmarks = preds['landmarks']
        b_landmarks = tf.reshape(b_landmarks,
                                 [batch_size, self.n_landmarks, 2])
        b_landmarks = tf.einsum('b c d, d ->b c d', b_landmarks,
                                self.resize_shape)
        b_landmarks = tf.einsum('b c d, b d ->b c d', b_landmarks,
                                self.resize_ratio)

        return b_landmarks
