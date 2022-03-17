from pprint import pprint
import tensorflow as tf


class CLSPostModel(tf.keras.Model):
    def __init__(self, pred_model, *args, **kwargs):
        super(CLSPostModel, self).__init__(*args, **kwargs)
        self.pred_model = pred_model

    @tf.function
    def call(self, x, training=False):
        imgs, _ = x
        rets = self.pred_model(imgs)
        return rets
