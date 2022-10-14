from pprint import pprint
from .base import Base
import numpy as np
import tensorflow as tf


class PosePostModel(tf.keras.Model):

    def __init__(self, pred_model, n_objs, top_k_n, kp_thres, nms_iou_thres,
                 resize_shape, *args, **kwargs):
        super(PosePostModel, self).__init__(*args, **kwargs)
        self.pred_model = pred_model
        self.n_objs = n_objs
        self.top_k_n = top_k_n
        self.kp_thres = kp_thres
        self.nms_iou_thres = nms_iou_thres
        self.resize_shape = tf.cast(resize_shape, tf.float32)
        self.base = Base()

    # @tf.function
    def call(self, x, training=False):
        imgs, origin_shapes = x
        batch_size = tf.shape(imgs)[0]
        self.resize_ratio = tf.cast(origin_shapes / self.resize_shape,
                                    tf.dtypes.float32)
        preds = self.pred_model(imgs, training=False)
        b_coors, b_params = self._obj_detect(batch_size, preds["obj_heat_map"],
                                             preds['obj_param_map'])
        return b_coors, b_params

    # @tf.function
    def _obj_detect(self, batch_size, hms, pms):
        hms = self.base.apply_max_pool(hms)
        b, h, w, c = [tf.shape(hms)[i] for i in range(4)]
        b_coors = self.base.top_k_loc(hms, self.top_k_n, h, w, c)
        # output = -tf.ones(shape=(batch_size, self.n_objs, c, 5))
        b_idxs = tf.tile(
            tf.range(0, b, dtype=tf.int32)[:, tf.newaxis, tf.newaxis,
                                           tf.newaxis],
            [1, c, self.top_k_n, 1],
        )
        b_infos = tf.concat([b_idxs, b_coors], axis=-1)
        b_params = tf.gather_nd(pms, b_infos)
        b_scores = tf.gather_nd(hms, b_infos)
        b_mask = tf.squeeze(b_scores > 0.5, axis=-1)
        b_params = b_params[b_mask]
        b_params = tf.reshape(b_params,
                              (batch_size, -1, tf.shape(b_params)[-1]))
        b_coors = tf.reshape(b_coors[b_mask],
                             (batch_size, -1, tf.shape(b_coors)[-1]))
        return b_coors, b_params
