from pprint import pprint
from .base import Base
from utils.io import load_BFM
import numpy as np
import tensorflow as tf


class SCRFDPostModel(tf.keras.Model):

    def __init__(self, pred_model, n_objs, top_k_n, kp_thres, nms_iou_thres,
                 resize_shape, *args, **kwargs):
        super(SCRFDPostModel, self).__init__(*args, **kwargs)
        self.pred_model = pred_model
        self.n_objs = n_objs
        self.top_k_n = top_k_n
        self.kp_thres = kp_thres
        self.nms_iou_thres = nms_iou_thres
        self.resize_shape = tf.cast(resize_shape, tf.float32)
        self.cls_out_channels = 1
        self._feat_stride_fpn = [8, 16, 32]
        self.num_levels = len(self._feat_stride_fpn)
        self.num_level_anchors = [3200, 800, 200]
        self._num_anchors = 2

    @tf.function
    def call(self, x, training=False):
        imgs, origin_shapes = x
        batch_size = tf.shape(imgs)[0]
        self.resize_ratio = tf.cast(origin_shapes / self.resize_shape,
                                    tf.dtypes.float32)
        preds = self.pred_model(imgs, training=False)
        box_results = self._anchor_assign(batch_size, preds["multi_lv_feats"])
        return box_results

    # @tf.function
    def _anchor_assign(self, batch_size, multi_lv_feats):
        b_outputs = -tf.ones(shape=(batch_size, self.n_objs,
                                    self.cls_out_channels, 5))
        obj_start_idx = 0
        for i, (lv_feats,
                stride) in enumerate(zip(multi_lv_feats,
                                         self._feat_stride_fpn)):
            b_cls_preds, b_bbox_preds = lv_feats

            b_cls_preds = tf.math.sigmoid(b_cls_preds)
            b_bbox_preds = tf.reshape(b_bbox_preds, [-1, 4])
            b_mask = b_cls_preds > self.kp_thres
            btach_idxs = tf.cast(tf.where(b_mask == True), tf.int32)[:, :1]
            b_cls_preds = tf.reshape(b_cls_preds, [-1, self.cls_out_channels])

            mask = b_cls_preds > self.kp_thres
            idxs = tf.where(mask == True)
            channel_idxs = tf.cast(idxs, tf.int32)[:, -1:]
            b_cls_preds = b_cls_preds[tf.math.reduce_any(mask, axis=-1)]
            b_bbox_preds = b_bbox_preds * stride
            height = self.resize_shape[0] // stride
            width = self.resize_shape[1] // stride
            X, Y = tf.meshgrid(tf.range(0, width), tf.range(0, height))
            anchor_centers = tf.stack([X, Y], axis=-1)
            anchor_centers = tf.reshape((anchor_centers * stride), (-1, 2))

            if self._num_anchors > 1:
                anchor_centers = tf.reshape(
                    tf.stack([anchor_centers] * self._num_anchors, axis=1),
                    (-1, 2))

            anchor_centers = tf.cast(anchor_centers, tf.float32)
            anchor_centers = tf.tile(anchor_centers[None, ...],
                                     (batch_size, 1, 1))
            anchor_centers = tf.reshape(anchor_centers, (-1, 2))

            b_bboxes = self.distance2bbox(anchor_centers, b_bbox_preds)
            b_bboxes = b_bboxes[tf.math.reduce_any(mask, axis=-1)]
            num_detected_objs = tf.math.reduce_sum(tf.cast(mask, tf.float32))
            obj_idxs = tf.range(num_detected_objs, dtype=tf.int32)[:, None]
            obj_idxs += obj_start_idx
            obj_start_idx += tf.cast(num_detected_objs, tf.int32)
            b_bboxes = tf.reshape(b_bboxes, (-1, 2, 2))
            b_bboxes = tf.einsum('n c d, b d -> n c d', b_bboxes[..., ::-1],
                                 self.resize_ratio)
            b_bboxes = tf.reshape(b_bboxes, (-1, 4))
            b_bboxes = tf.concat([b_bboxes, b_cls_preds], axis=-1)
            idxs = tf.concat([btach_idxs, obj_idxs, channel_idxs], axis=-1)
            b_outputs = tf.tensor_scatter_nd_update(b_outputs, idxs, b_bboxes)

        b_scores = b_outputs[..., -1]
        b_outputs = b_outputs[..., :-1]
        # [B, N, Cate, 4]
        nms_reuslt = tf.image.combined_non_max_suppression(
            b_outputs,
            b_scores,
            self.n_objs,
            self.n_objs,
            iou_threshold=self.nms_iou_thres,
            clip_boxes=False)
        box_results = tf.where(nms_reuslt[0] == -1., np.inf, nms_reuslt[0])
        box_results = tf.where((box_results - 1.) == -1., np.inf, box_results)
        b_bboxes = tf.concat(
            [box_results, nms_reuslt[1][..., None], nms_reuslt[2][..., None]],
            axis=-1)
        b_bboxes = tf.where(b_bboxes == -1., np.inf, b_bboxes)
        b_bboxes = tf.reshape(b_bboxes, [-1, self.n_objs, 6])
        return b_bboxes

    def distance2bbox(self, points, distance, max_shape=None):
        """Decode distance prediction to bounding box.
        Args:
            points (Tensor): Shape (n, 2), [x, y].
            distance (Tensor): Distance from the given point to 4
                boundaries (left, top, right, bottom).
            max_shape (tuple): Shape of the image.

        Returns:
            Tensor: Decoded bboxes.
        """
        x1 = points[..., 0] - distance[..., 0]
        y1 = points[..., 1] - distance[..., 1]
        x2 = points[..., 0] + distance[..., 2]
        y2 = points[..., 1] + distance[..., 3]
        if max_shape is not None:
            x1 = tf.clip_by_value(x1,
                                  clip_value_min=0,
                                  clip_value_max=max_shape[1])
            y1 = tf.clip_by_value(y1,
                                  clip_value_min=0,
                                  clip_value_max=max_shape[0])
            x2 = tf.clip_by_value(x2,
                                  clip_value_min=0,
                                  clip_value_max=max_shape[1])
            y2 = tf.clip_by_value(y2,
                                  clip_value_min=0,
                                  clip_value_max=max_shape[0])
        return tf.stack([x1, y1, x2, y2], axis=-1)
