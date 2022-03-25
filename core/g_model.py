from pprint import pprint
from .base import Base
import numpy as np
import tensorflow as tf


class GPostModel(tf.keras.Model):
    def __init__(self, pred_model, n_objs, top_k_n, kp_thres, nms_iou_thres,
                 resize_shape, *args, **kwargs):
        super(GPostModel, self).__init__(*args, **kwargs)
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

        heatmap_y, heatmap_x = preds["heatmap_y"], preds["heatmap_x"]
        heatmap = tf.linalg.matmul(heatmap_y[..., tf.newaxis],
                                   heatmap_x[..., tf.newaxis],
                                   transpose_b=True,
                                   output_type=tf.float32)
        b_bboxes = self._obj_detect(batch_size, heatmap[..., tf.newaxis],
                                    preds['obj_size_maps'])
        return b_bboxes

    # @tf.function
    def _obj_detect(self, batch_size, hms, size_maps):
        hms = self.base.apply_max_pool(hms)
        b, h, w, c = [tf.shape(hms)[i] for i in range(4)]
        b_coors = self.base.top_k_loc(hms, self.top_k_n, h, w, c)
        b_coors = b_coors[:, :1, ...]
        output = -tf.ones(shape=(batch_size, self.n_objs, c, 5))

        b_idxs = tf.tile(
            tf.range(0, b, dtype=tf.int32)[:, tf.newaxis, tf.newaxis,
                                           tf.newaxis],
            [1, c, self.top_k_n, 1],
        )

        b_infos = tf.concat([b_idxs, b_coors], axis=-1)
        # only pick bbox

        b_size_vals = tf.gather_nd(size_maps, b_infos)

        b_c_idxs = tf.tile(
            tf.range(0, c, dtype=tf.int32)[tf.newaxis, :, tf.newaxis,
                                           tf.newaxis],
            [b, 1, self.top_k_n, 1])

        b_infos = tf.concat([b_infos, b_c_idxs], axis=-1)
        b_scores = tf.gather_nd(hms, b_infos)

        b_centers = tf.cast(b_coors, tf.float32)
        b_tls = (b_centers - b_size_vals / 2)
        b_brs = (b_centers + b_size_vals / 2)
        # clip value
        b_br_y = b_brs[..., 0]
        b_br_x = b_brs[..., 1]
        b_tls = tf.where(b_tls < 0., 0., b_tls)

        b_br_y = tf.where(b_brs[..., :1] > self.resize_shape[0] - 1.,
                          self.resize_shape[0] - 1., b_brs[..., :1])
        b_br_x = tf.where(b_brs[..., -1:] > self.resize_shape[1] - 1.,
                          self.resize_shape[1] - 1., b_brs[..., -1:])
        b_brs = tf.concat([b_br_y, b_br_x], axis=-1)

        b_bboxes = tf.concat([b_tls, b_brs], axis=-1)

        b_bboxes = self.base.resize_back(b_bboxes, self.resize_ratio)

        b_scores = tf.transpose(b_scores, [0, 2, 1])
        # B N C D
        b_bboxes = tf.concat([b_bboxes, b_scores[..., None]], axis=-1)

        mask = b_scores > self.kp_thres
        index = tf.where(mask == True)
        n = tf.shape(index)[0]
        d = tf.shape(b_bboxes)[-1]
        c_idx = tf.tile(tf.range(d)[None, :, None], [n, 1, 1])
        index = tf.cast(tf.tile(index[:, tf.newaxis, :], [1, d, 1]), tf.int32)
        index = tf.concat([index, c_idx], axis=-1)
        output = tf.tensor_scatter_nd_update(output, index, b_bboxes[mask])
        scores = output[..., -1]
        output = output[..., :-1]
        # [B, N, Cate, 4]
        nms_reuslt = tf.image.combined_non_max_suppression(
            output,
            scores,
            self.n_objs,
            self.n_objs,
            iou_threshold=self.nms_iou_thres,
            clip_boxes=False)
        box_results = tf.where(nms_reuslt[0] == -1., np.inf, nms_reuslt[0])
        box_results = tf.where((box_results - 1.) == -1., np.inf, box_results)
        b_bboxes = tf.concat(
            [box_results, nms_reuslt[1][..., None], nms_reuslt[2][..., None]],
            axis=-1)
        b_bboxes = tf.reshape(b_bboxes, [-1, self.n_objs, 6])
        return b_bboxes
