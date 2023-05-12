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

        self.feat_size = tf.constant([(80, 80), (40, 40), (20, 20)])
        self.reg_max = 8
        self.cls_out_channels = 1
        self.fmc = 3
        self._feat_stride_fpn = [8, 16, 32]
        self._num_anchors = 2
        self.use_kps = True

    # @tf.function
    def call(self, x, training=False):
        imgs, origin_shapes = x
        batch_size = tf.shape(imgs)[0]
        self.resize_ratio = tf.cast(origin_shapes / self.resize_shape,
                                    tf.dtypes.float32)
        preds = self.pred_model(imgs, training=False)
        box_results, kp_results = self._anchor_assign(batch_size,
                                                      preds["multi_lv_feats"])
        return box_results, kp_results

    # @tf.function
    def _anchor_assign(self, batch_size, multi_lv_feats):
        boxes_list, kp_list = [], []
        output = -tf.ones(shape=(batch_size, self.n_objs, 1, 5))
        for i, (lv_feats,
                stride) in enumerate(zip(multi_lv_feats,
                                         self._feat_stride_fpn)):
            b_cls_preds, b_bbox_preds, b_kp_preds = lv_feats
            b_bbox_preds = tf.reshape(b_bbox_preds, [-1, 4])
            b_cls_preds = tf.reshape(b_cls_preds, [-1, self.cls_out_channels])
            b_cls_preds = tf.math.sigmoid(b_cls_preds)
            mask = tf.squeeze(b_cls_preds > 0.05, axis=-1)
            pos_ind = tf.where(mask == True)
            b_cls_preds = b_cls_preds[mask]
            b_bbox_preds = b_bbox_preds * stride
            b_kp_preds = tf.reshape(b_kp_preds, [-1, 5 * 2]) * stride
            height = self.resize_shape[0] // stride
            width = self.resize_shape[1] // stride
            X, Y = tf.meshgrid(tf.range(width), tf.range(height))
            anchor_centers = tf.stack([X, Y], axis=-1)
            anchor_centers = tf.reshape((anchor_centers * stride), (-1, 2))
            b_bbox_preds = tf.reshape(b_bbox_preds, [batch_size, -1, 4])
            if self._num_anchors > 1:
                anchor_centers = tf.reshape(
                    tf.stack([anchor_centers] * self._num_anchors, axis=1),
                    (-1, 2))
            anchor_centers = tf.cast(anchor_centers, tf.float32)
            b_bboxes = self.distance2bbox(anchor_centers, b_bbox_preds)
            mask = tf.tile(mask[None, :], [batch_size, 1])
            b_bboxes = b_bboxes[mask]
            boxes_list.append(b_bboxes)
            kpss = self.distance2kps(anchor_centers, b_kp_preds)
            kpss = tf.reshape(kpss, (batch_size, tf.shape(kpss)[0], -1, 2))
            pos_kpss = kpss[mask]
            kp_list.append(pos_kpss)
        b_bboxes = tf.reshape(tf.concat(b_bboxes, axis=0),
                              [batch_size, -1, 2, 2])
        b_bboxes = tf.einsum('b n c d, b d -> b n c d', b_bboxes,
                             self.resize_ratio[:, ::-1])
        kp_list = tf.reshape(tf.concat(kp_list, axis=0), [batch_size, -1, 5, 2])
        kp_list = tf.einsum('b n c d, b d -> b n c d', kp_list,
                            self.resize_ratio[:, ::-1])
        return b_bboxes, kp_list

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
            y1 = tf.clip_by_value(x1,
                                  clip_value_min=0,
                                  clip_value_max=max_shape[0])
            x2 = tf.clip_by_value(x1,
                                  clip_value_min=0,
                                  clip_value_max=max_shape[1])
            y2 = tf.clip_by_value(x1,
                                  clip_value_min=0,
                                  clip_value_max=max_shape[0])
        return tf.stack([x1, y1, x2, y2], axis=-1)

    def distance2kps(self, points, distance, max_shape=None):
        """Decode distance prediction to bounding box.

        Args:
            points (Tensor): Shape (n, 2), [x, y].
            distance (Tensor): Distance from the given point to 4
                boundaries (left, top, right, bottom).
            max_shape (tuple): Shape of the image.

        Returns:
            Tensor: Decoded bboxes.
        """
        preds = []
        for i in range(0, 10, 2):
            px = points[:, i % 2] + distance[:, i]
            py = points[:, i % 2 + 1] + distance[:, i + 1]
            if max_shape is not None:
                px = tf.clip_by_value(px,
                                      clip_value_min=0,
                                      clip_value_max=max_shape[1])
                py = tf.clip_by_value(py,
                                      clip_value_min=0,
                                      clip_value_max=max_shape[1])
            preds.append(px)
            preds.append(py)
        return tf.stack(preds, axis=-1)