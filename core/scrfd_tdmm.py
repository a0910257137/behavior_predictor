from pprint import pprint
from .base import Base
from utils.io import load_BFM
import numpy as np
import tensorflow as tf
from math import cos, sin
import os


class SCRFDTDMMPostModel(tf.keras.Model):

    def __init__(self, tdmm_cfg, pred_model, n_objs, top_k_n, kp_thres,
                 nms_iou_thres, resize_shape, *args, **kwargs):
        super(SCRFDTDMMPostModel, self).__init__(*args, **kwargs)
        self.pred_model = pred_model
        self.n_objs = n_objs
        self.top_k_n = top_k_n
        self.kp_thres = kp_thres
        self.nms_iou_thres = nms_iou_thres
        self.resize_shape = tf.cast(resize_shape, tf.float32)
        self.cls_out_channels = 2
        # self._feat_stride_fpn = [8, 16, 32]
        self._feat_stride_fpn = [16]
        self.num_levels = len(self._feat_stride_fpn)
        self.num_level_anchors = [3200, 800, 200]
        self._num_anchors = 2
        self.n_R = tdmm_cfg['n_R']
        self.n_shp = tdmm_cfg['n_shp']
        self.n_exp = tdmm_cfg['n_exp']
        pms = tf.cast(np.load(tdmm_cfg['pms_path']), tf.float32)
        pms_R = pms[:, :self.n_R]
        pms_shp = pms[:, self.n_R:self.n_R + self.n_shp]
        pms_exp = pms[:, self.n_R + self.n_shp:-3]
        pms = tf.concat([pms_R, pms_shp, pms_exp], axis=-1)
        self.pms = pms[:2, :]
        head_model = load_BFM(tdmm_cfg['model_path'])
        kpt_ind = head_model['kpt_ind']
        X_ind_all = np.stack([kpt_ind * 3, kpt_ind * 3 + 1, kpt_ind * 3 + 2])
        X_ind_all = tf.concat([
            X_ind_all[:, :17], X_ind_all[:, 17:27], X_ind_all[:, 36:48],
            X_ind_all[:, 27:36], X_ind_all[:, 48:68]
        ],
                              axis=-1)
        valid_ind = tf.reshape(tf.transpose(X_ind_all), (-1))
        self.valid_ind = tf.cast(valid_ind, tf.int32)
        self.u_base = tf.cast(head_model['shapeMU'], tf.float32)
        self.u_base = tf.gather(self.u_base, self.valid_ind)
        self.u_base = tf.reshape(self.u_base,
                                 (tf.shape(self.u_base)[0] // 3, 3))
        self.u_base = tf.reshape(self.u_base,
                                 (tf.shape(self.u_base)[0] * 3, 1))
        self.shp_base = tf.cast(head_model['shapePC'],
                                tf.float32)[:, :self.n_shp]
        self.shp_base = tf.gather(self.shp_base, self.valid_ind)
        self.exp_base = tf.cast(head_model['expPC'],
                                tf.float32)[:, :self.n_exp]
        self.exp_base = tf.gather(self.exp_base, self.valid_ind)

    # @tf.function
    def call(self, x, training=False):
        imgs, origin_shapes = x
        batch_size = tf.shape(imgs)[0]
        self.resize_ratio = tf.cast(origin_shapes / self.resize_shape,
                                    tf.dtypes.float32)
        preds = self.pred_model(imgs, training=False)
        box_results, b_lnmk_outputs = self._anchor_assign(
            batch_size, preds["multi_lv_feats"])

        return box_results, b_lnmk_outputs

    # @tf.function
    def _anchor_assign(self, batch_size, multi_lv_feats):
        b_outputs = -tf.ones(shape=(batch_size, self.n_objs,
                                    self.cls_out_channels, 5))
        b_lnmk_outputs = tf.ones(shape=(batch_size, self.n_objs,
                                        self.cls_out_channels, 68, 2)) * np.inf
        obj_start_idx = 0
        b_idx_list, b_bbox_list, b_lnmk_list = [], [], []
        b_params_list, b_kpss_list = [], []
        for i, (lv_feats,
                stride) in enumerate(zip(multi_lv_feats,
                                         self._feat_stride_fpn)):
            # if i == 0:
            #     continue
            b_cls_preds, b_bbox_preds, b_param_preds, b_param_kps = lv_feats
            b_cls_preds = tf.math.sigmoid(b_cls_preds)
            b_bbox_preds = tf.reshape(b_bbox_preds, [-1, 4])
            # b_param_preds = tf.reshape(b_param_preds, [-1] + map_size +
            #                            [self.n_R + self.n_shp + self.n_exp])
            b_param_preds = tf.reshape(
                b_param_preds, (-1, self.n_R + self.n_shp + self.n_exp))
            b_param_kps = tf.reshape(b_param_kps, (-1, 2))
            b_mask = b_cls_preds > self.kp_thres
            b_map_idxs = tf.cast(tf.where(b_mask == True), tf.int32)
            btach_idxs = b_map_idxs[:, :1]
            b_cls_preds = tf.reshape(b_cls_preds, [-1, self.cls_out_channels])
            mask = b_cls_preds > self.kp_thres
            idxs = tf.where(mask == True)
            channel_idxs = tf.cast(idxs, tf.int32)[:, -1:]
            b_cls_preds = tf.reshape(tf.gather_nd(b_cls_preds, idxs), [-1, 1])

            b_bbox_preds = b_bbox_preds * stride
            b_bbox_preds = tf.gather_nd(b_bbox_preds, idxs[:, :1])
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
            anchor_centers = tf.gather_nd(anchor_centers, idxs[:, :1])
            b_bboxes = self.distance2bbox(anchor_centers, b_bbox_preds)
            b_param_preds = tf.gather_nd(b_param_preds, idxs[:, :1])
            # b_param_preds = tf.gather_nd(b_param_preds, b_map_idxs[:, :3])
            # b_param_kps = tf.gather_nd(b_param_kps, b_map_idxs[:, :3])

            b_param_kps = tf.gather_nd(b_param_kps, idxs[:, :1])
            b_param_kps *= stride
            b_param_kps = self.distance2kps(anchor_centers, b_param_kps)
            b_param_preds = b_param_preds * self.pms[1] + self.pms[0]
            b_params_list.append(b_param_preds)
            vertices = self.u_base[None, ...] + tf.linalg.matmul(
                self.shp_base, b_param_preds[:, self.n_R:self.n_R + self.n_shp,
                                             None]
            ) + tf.linalg.matmul(
                self.exp_base, b_param_preds[:, self.n_R + self.n_shp:, None])
            vertices = tf.reshape(vertices,
                                  (-1, tf.shape(vertices)[-2] // 3, 3))
            b_R = tf.reshape(b_param_preds[:, :self.n_R], [-1, 3, 3])
            b_lnmks = tf.linalg.matmul(vertices, b_R, transpose_b=(0, 2, 1))
            num_detected_objs = tf.math.reduce_sum(tf.cast(mask, tf.float32))
            obj_idxs = tf.range(num_detected_objs, dtype=tf.int32)[:, None]
            obj_idxs += obj_start_idx
            obj_start_idx += tf.cast(num_detected_objs, tf.int32)
            b_bboxes = tf.reshape(b_bboxes, (-1, 2, 2))
            b_bboxes = tf.einsum('n c d, b d -> n c d', b_bboxes[..., ::-1],
                                 self.resize_ratio)
            b_bboxes = tf.reshape(b_bboxes, (-1, 4))
            b_bbox_list.append(b_bboxes)
            b_bboxes = tf.concat([b_bboxes, b_cls_preds], axis=-1)
            idxs = tf.concat([btach_idxs, obj_idxs, channel_idxs], axis=-1)
            b_idx_list.append(idxs)
            b_lnmk_list.append(b_lnmks[:, :, :2][..., ::-1])
            b_param_kps = tf.concat([b_param_kps[:, :1], b_param_kps[:, 1:]],
                                    axis=-1)
            b_param_kps = tf.einsum('n d, b d -> n d',
                                    b_param_kps[:, ::-1] + 0.5,
                                    self.resize_ratio)
            b_kpss_list.append(b_param_kps)
            b_outputs = tf.tensor_scatter_nd_update(b_outputs, idxs, b_bboxes)
        b_idx_list = tf.concat(b_idx_list, axis=0)
        b_bbox_list = tf.concat(b_bbox_list, axis=0)
        b_lnmk_list = tf.concat(b_lnmk_list, axis=0)
        b_params_list = tf.concat(b_params_list, axis=0)
        b_kpss_list = tf.concat(b_kpss_list, axis=0)
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
        search_bboxes = tf.reshape(
            box_results, (-1, 4))[:, None, :] - b_bbox_list[None, :, :]
        search_masks = tf.math.reduce_all(search_bboxes == 0., axis=-1)
        search_idxs = tf.where(search_masks == True)
        b_idx_list = tf.gather_nd(b_idx_list, search_idxs[:, -1:])
        b_bbox_list = tf.gather_nd(b_bbox_list, search_idxs[:, -1:])
        b_bbox_tls, b_bbox_brs = b_bbox_list[..., :2], b_bbox_list[..., 2:4]
        b_lnmk_list = tf.gather_nd(b_lnmk_list, search_idxs[:, -1:])
        b_kpss_list = tf.gather_nd(b_kpss_list, search_idxs[:, -1:])
        b_params_list = tf.gather_nd(b_params_list, search_idxs[:, -1:])
        b_lnmk_tls, b_lnmk_brs = tf.math.reduce_min(
            b_lnmk_list, axis=-2), tf.math.reduce_max(b_lnmk_list, axis=-2)
        b_bbox_wh = b_bbox_brs - b_bbox_tls
        b_lnmk_wh = b_lnmk_brs - b_lnmk_tls
        b_scales = b_bbox_wh / b_lnmk_wh
        b_lnmk_list = tf.einsum('n c d, n d -> n c d', b_lnmk_list, b_scales)
        # b_lnmk_tls, b_lnmk_brs = tf.math.reduce_min(
        #     b_lnmk_list, axis=-2), tf.math.reduce_max(b_lnmk_list, axis=-2)
        # b_bbox_centers = (b_bbox_tls + b_bbox_brs) / 2
        # b_lnmk_centers = (b_lnmk_tls + b_lnmk_brs) / 2
        # b_shifting = b_bbox_centers - b_lnmk_centers
        # b_params_list = tf.concat([b_scales, b_params_list, b_kpss_list],
        #                           axis=-1)
        b_lnmk_list = b_lnmk_list + b_kpss_list[:, None, :]
        b_lnmk_outputs = tf.tensor_scatter_nd_update(b_lnmk_outputs,
                                                     b_idx_list, b_lnmk_list)
        b_bboxes = tf.concat(
            [box_results, nms_reuslt[1][..., None], nms_reuslt[2][..., None]],
            axis=-1)
        b_bboxes = tf.where(b_bboxes == -1., np.inf, b_bboxes)
        b_bboxes = tf.reshape(b_bboxes, [-1, self.n_objs, 6])
        return b_bboxes, b_lnmk_outputs

    @tf.function
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

    @tf.function
    def distance2kps(self, points, distance):
        """Decode distance prediction to bounding box.

        Args:
            points (Tensor): Shape (n, 2), [x, y].
            distance (Tensor): Distance from the given point to 4
                boundaries (left, top, right, bottom).
            max_shape (tuple): Shape of the image.

        Returns:
            Tensor: Decoded bboxes.
        """
        C = 2
        preds = []
        for i in range(0, C, 2):
            px = points[:, i % 2] + distance[:, i]
            py = points[:, i % 2 + 1] + distance[:, i + 1]
            preds.append(px)
            preds.append(py)
        return tf.stack(preds, axis=-1)
