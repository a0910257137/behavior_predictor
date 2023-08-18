from pprint import pprint
from .base import Base
from utils.io import load_BFM
import numpy as np
import tensorflow as tf


class SCRFDTDMMPostModel(tf.keras.Model):

    def __init__(self, tdmm_cfg, pred_model, n_objs, top_k_n, kp_thres,
                 nms_iou_thres, resize_shape, *args, **kwargs):
        super(SCRFDTDMMPostModel, self).__init__(*args, **kwargs)
        self.n_R = tdmm_cfg['n_R']
        self.n_shp, self.n_exp = tdmm_cfg['n_shp'], tdmm_cfg['n_exp']
        pms = tf.cast(np.load(tdmm_cfg['pms_path']), tf.float32)
        pms_R = pms[:, :self.n_R]
        pms_shp, pms_exp = pms[:, self.n_R:self.n_R + self.n_shp], pms[:,
                                                                       208:-3]
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
        self.u_base = tf.reshape(self.u_base, (tf.shape(self.u_base)[0] * 3, 1))
        self.shp_base = tf.cast(head_model['shapePC'],
                                tf.float32)[:, :self.n_shp]
        self.shp_base = tf.gather(self.shp_base, self.valid_ind)
        self.exp_base = tf.cast(head_model['expPC'], tf.float32)
        self.exp_base = tf.gather(self.exp_base, self.valid_ind)
        self.pred_model = pred_model
        self.n_objs = n_objs
        self.top_k_n = top_k_n
        self.kp_thres = kp_thres
        self.nms_iou_thres = nms_iou_thres
        self.resize_shape = tf.cast(resize_shape, tf.float32)
        self.cls_out_channels = 2
        self._feat_stride_fpn = [8, 16, 32]
        self.num_levels = len(self._feat_stride_fpn)
        self.num_level_anchors = [3200, 800, 200]
        self._num_anchors = 2

    def call(self, x, training=False):
        imgs, origin_shapes = x
        batch_size = tf.shape(imgs)[0]
        self.resize_ratio = tf.cast(origin_shapes / self.resize_shape,
                                    tf.dtypes.float32)
        preds = self.pred_model(imgs, training=False)
        box_results, lnms_results = self._anchor_assign(batch_size,
                                                        preds["multi_lv_feats"])
        return box_results, lnms_results

    # @tf.function
    def _anchor_assign(self, batch_size, multi_lv_feats):
        b_bbox_outputs = -tf.ones(shape=(batch_size, self.n_objs,
                                         self.cls_out_channels, 5))
        b_lnmk_outputs = -tf.ones(shape=(batch_size, self.n_objs,
                                         self.cls_out_channels, 68, 2))
        obj_start_idx = 0
        bbox_list, lnmk_list = [], []
        idxs_list = []
        for i, (lv_feats,
                stride) in enumerate(zip(multi_lv_feats,
                                         self._feat_stride_fpn)):
            if i == 0:
                continue
            b_cls_preds, b_bbox_preds, b_param_preds = lv_feats
            b_cls_preds = tf.math.sigmoid(b_cls_preds)

            b_bbox_preds = tf.reshape(b_bbox_preds, [-1, 4])
            b_param_preds = tf.reshape(b_param_preds,
                                       [-1, self.n_R + self.n_shp + self.n_exp])
            b_mask = b_cls_preds > self.kp_thres
            btach_idxs = tf.cast(tf.where(b_mask == True), tf.int32)[:, :1]
            b_cls_preds = tf.reshape(b_cls_preds, [-1, self.cls_out_channels])
            mask = b_cls_preds > self.kp_thres
            idxs = tf.where(mask == True)
            channel_idxs = tf.cast(idxs, tf.int32)[:, -1:]
            b_cls_preds = tf.expand_dims(b_cls_preds[mask], axis=-1)
            b_bboxes = self.decode_bbox(batch_size, stride, idxs, b_bbox_preds)
            pred_R, pred_shp, pred_exp = self.decod_params(idxs, b_param_preds)
            n_lnmks = self.reconstruct_lnmks(batch_size, b_bboxes, pred_R,
                                             pred_shp, pred_exp)
            num_detected_objs = tf.math.reduce_sum(tf.cast(mask, tf.float32))
            obj_idxs = tf.range(num_detected_objs, dtype=tf.int32)[:, None]
            obj_idxs += obj_start_idx
            b_bboxes = tf.einsum('n c d, b d -> n c d', b_bboxes[..., ::-1],
                                 self.resize_ratio)
            b_bboxes = tf.reshape(b_bboxes, (-1, 4))
            b_bboxes = tf.concat([b_bboxes, b_cls_preds], axis=-1)
            idxs = tf.concat([btach_idxs, obj_idxs, channel_idxs], axis=-1)
            n_lnmks = tf.einsum('n c d, b d -> n c d', n_lnmks[..., ::-1],
                                self.resize_ratio)
            bbox_list.append(b_bboxes[:, :-1])
            lnmk_list.append(n_lnmks)
            idxs_list.append(idxs)
            b_bbox_outputs = tf.tensor_scatter_nd_update(
                b_bbox_outputs, idxs, b_bboxes)
        bbox_tensor = tf.concat(bbox_list, axis=0)
        lnmk_tensor = tf.concat(lnmk_list, axis=0)
        idxs_tensor = tf.concat(idxs_list, axis=0)
        b_scores = b_bbox_outputs[..., -1]
        b_bbox_outputs = b_bbox_outputs[..., :-1]
        # [B, N, Cate, 4]
        nms_reuslt = tf.image.combined_non_max_suppression(
            b_bbox_outputs,
            b_scores,
            self.n_objs,
            self.n_objs,
            iou_threshold=self.nms_iou_thres,
            clip_boxes=False)
        box_results = tf.where(nms_reuslt[0] == -1., np.inf, nms_reuslt[0])

        search_tensors = tf.reshape(
            box_results, [-1, 4])[:, None, :] - bbox_tensor[None, :, :]
        search_mask = tf.math.reduce_all(search_tensors == 0.0, axis=-1)
        idxs = tf.where(search_mask == True)[:, -1:]
        lnmk_tensor = tf.gather_nd(lnmk_tensor, idxs)
        idxs_tensor = tf.gather_nd(idxs_tensor, idxs)

        box_results = tf.where((box_results - 1.) == -1., np.inf, box_results)
        b_bboxes = tf.concat(
            [box_results, nms_reuslt[1][..., None], nms_reuslt[2][..., None]],
            axis=-1)
        b_bboxes = tf.where(b_bboxes == -1., np.inf, b_bboxes)
        b_bboxes = tf.reshape(b_bboxes, [-1, self.n_objs, 6])

        b_lnmk_outputs = tf.tensor_scatter_nd_update(b_lnmk_outputs,
                                                     idxs_tensor, lnmk_tensor)
        b_lnmk_outputs = tf.where(b_lnmk_outputs == -1., np.inf, b_lnmk_outputs)
        return b_bboxes, b_lnmk_outputs

    def decod_params(self, idxs, b_param_preds):
        b_param_preds = tf.gather_nd(b_param_preds, idxs[:, 0][:, None])
        b_param_preds = b_param_preds * self.pms[1][None, :] + self.pms[0][
            None, :]
        R = b_param_preds[:, :self.n_R]
        shp = b_param_preds[:, self.n_R:self.n_R + self.n_shp]
        exp = b_param_preds[:, self.n_R + self.n_shp:]
        return R, shp, exp

    def decode_bbox(self, batch_size, stride, idxs, b_bbox_preds):
        b_bbox_preds = b_bbox_preds * stride
        height = self.resize_shape[0] // stride
        width = self.resize_shape[1] // stride
        X, Y = tf.meshgrid(tf.range(0, width), tf.range(0, height))
        anchor_centers = tf.stack([X, Y], axis=-1)
        anchor_centers = tf.reshape((anchor_centers * stride), (-1, 2))

        if self._num_anchors > 1:
            anchor_centers = tf.reshape(
                tf.stack([anchor_centers] * self._num_anchors, axis=1), (-1, 2))

        anchor_centers = tf.cast(anchor_centers, tf.float32)
        anchor_centers = tf.tile(anchor_centers[None, ...], (batch_size, 1, 1))
        anchor_centers = tf.reshape(anchor_centers, (-1, 2))
        b_bboxes = self.distance2bbox(anchor_centers, b_bbox_preds)
        b_bboxes = tf.gather_nd(b_bboxes, idxs[:, :1])
        b_bboxes = tf.reshape(b_bboxes, (-1, 2, 2))
        return b_bboxes

    def reconstruct_lnmks(self, batch_size, b_bboxes, R, shp, exp):
        n_lnmks = self.u_base + tf.linalg.matmul(
            self.shp_base, shp[..., None]) + tf.linalg.matmul(
                self.exp_base, exp[..., None])
        n_lnmks = tf.reshape(n_lnmks, (-1, tf.shape(n_lnmks)[-2] // 3, 3))
        R = tf.reshape(R, [-1, 3, 3])
        n_lnmks = tf.linalg.matmul(n_lnmks, R, transpose_b=(0, 2, 1))
        n_lnmks = n_lnmks[..., :2]
        n_lnmk_tls = tf.math.reduce_min(n_lnmks, axis=-2, keepdims=True)
        n_lnmk_brs = tf.math.reduce_max(n_lnmks, axis=-2, keepdims=True)
        n_bbox_tls = b_bboxes[:, :1, :]
        n_bbox_brs = b_bboxes[:, 1:, :]
        n_lnmks_wh = n_lnmk_brs - n_lnmk_tls
        n_bbox_wh = n_bbox_brs - n_bbox_tls
        n_scales = n_bbox_wh / n_lnmks_wh
        n_lnmks = tf.math.abs(n_scales) * n_lnmks
        return n_lnmks[..., :2]

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
