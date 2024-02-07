from pprint import pprint
from .base import Base
from utils.io import load_BFM
import numpy as np
import tensorflow as tf
import os
from glob import glob


class SCRFDTFLiteOptModel(tf.keras.Model):

    def __init__(self, tdmm_cfg, weight_root_dir, interpreter, n_objs, top_k_n,
                 kp_thres, nms_iou_thres, resize_shape, *args, **kwargs):
        super(SCRFDTFLiteOptModel, self).__init__(*args, **kwargs)

        def _get_bbox_weights(bbox_dir):
            bbox_weight_paths = list(glob(os.path.join(bbox_dir, "*.npy")))
            bbox_weight_paths = sorted(bbox_weight_paths)
            bbox_weights, bbox_biass, bbox_scales = [], [], []
            for path in bbox_weight_paths:
                vals = tf.cast(np.load(path), tf.float32)
                if 'scale_bbox' in path:
                    bbox_scales.append(vals)
                elif ('bias_pred' in path):
                    bbox_biass.append(vals)
                elif 'weight_pred' in path:
                    vals = np.transpose(vals, axes=(0, 1, 3, 2))
                    bbox_weights.append(vals)
            return bbox_weights, bbox_biass, bbox_scales

        def _get_param_kp(weight_dir):
            paths = list(glob(os.path.join(weight_dir, "*.npy")))
            paths = sorted(paths)
            weighs_conv3x3, biass_conv3x3, weights_pred, biass_pred = [], [], [], []
            for path in paths:
                vals = tf.cast(np.load(path), tf.float32)
                if 'bias_conv3x3' in path:
                    biass_conv3x3.append(vals)
                elif 'bias_pred' in path:
                    biass_pred.append(vals)
                elif 'weight_conv3x3' in path:
                    vals = np.transpose(vals, axes=(0, 1, 3, 2))
                    weighs_conv3x3.append(vals)
                elif 'weight_pred' in path:
                    vals = np.transpose(vals, axes=(0, 1, 3, 2))
                    weights_pred.append(vals)
            return weighs_conv3x3, biass_conv3x3, weights_pred, biass_pred

        self.interpreter = interpreter
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.map_keys = [
            "stride_2_x",
            "stride_1_cls",
            "stride_0_x",
            "stride_0_cls",
            "stride_1_x",
            "stride_2_bbox_x",
            "stride_2_cls",
            "stride_0_bbox_x",
            "stride_1_bbox_x",
        ]
        self.multi_lvs_keys = [[
            "stride_0_cls",
            "stride_0_bbox_x",
            "stride_0_x",
        ], [
            "stride_1_cls",
            "stride_1_bbox_x",
            "stride_1_x",
        ], [
            "stride_2_cls",
            "stride_2_bbox_x",
            "stride_2_x",
        ]]
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
        self.n_R = tdmm_cfg['n_R']
        self.n_shp = tdmm_cfg['n_shp']
        self.n_exp = tdmm_cfg['n_exp']
        pms = tf.cast(np.load(tdmm_cfg['pms_path']), tf.float32)
        pms_R = pms[:, :self.n_R]
        pms_shp = pms[:, self.n_R:self.n_R + self.n_shp]
        pms_exp = pms[:, self.n_R + self.n_shp:-3]
        pms = tf.concat([pms_R, pms_shp, pms_exp], axis=-1)
        self.pms = pms[:2, :]
        # self.pms.numpy().reshape([-1]).astype(np.float32).tofile(
        #     "/aidata/anders/data_collection/okay/total/archives/whole/scale_down/BFM/pms.bin"
        # )
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
        # self.u_base.numpy().reshape([-1]).astype(np.float32).tofile(
        #     "/aidata/anders/data_collection/okay/total/archives/whole/scale_down/BFM/u_base.bin"
        # )
        self.u_base = tf.reshape(self.u_base,
                                 (tf.shape(self.u_base)[0] // 3, 3))
        self.u_base = tf.reshape(self.u_base,
                                 (tf.shape(self.u_base)[0] * 3, 1))
        self.shp_base = tf.cast(head_model['shapePC'],
                                tf.float32)[:, :self.n_shp]
        self.shp_base = tf.gather(self.shp_base, self.valid_ind)
        # self.shp_base.numpy().reshape([-1]).astype(np.float32).tofile(
        #     "/aidata/anders/data_collection/okay/total/archives/whole/scale_down/BFM/shp_base.bin"
        # )
        self.exp_base = tf.cast(head_model['expPC'],
                                tf.float32)[:, :self.n_exp]
        self.exp_base = tf.gather(self.exp_base, self.valid_ind)
        # self.exp_base.numpy().reshape([-1]).astype(np.float32).tofile(
        #     "/aidata/anders/data_collection/okay/total/archives/whole/scale_down/BFM/exp_base.bin"
        # )
        bbox_dir = os.path.join(weight_root_dir, "bbox")
        kps_dir = os.path.join(weight_root_dir, "kps")
        params_dir = os.path.join(weight_root_dir, "params")
        self.bbox_weights, self.bbox_biass, self.bbox_scales = _get_bbox_weights(
            bbox_dir)
        self.param_weighs_conv3x3, self.param_biass_conv3x3, self.param_weights_pred, self.param_biass_pred = _get_param_kp(
            params_dir)
        self.kp_weighs_conv3x3, self.kp_biass_conv3x3, self.kp_weights_pred, self.kp_biass_pred = _get_param_kp(
            kps_dir)
        X, Y = tf.meshgrid(tf.range(-1, 2), tf.range(-1, 2))
        self.grid_3x3 = tf.stack([Y, X], axis=-1)
        X, Y = tf.meshgrid(tf.range(-2, 3), tf.range(-2, 3))
        self.grid_5x5 = tf.stack([Y, X], axis=-1)

    # @tf.function
    def call(self, x, training=False):
        imgs, origin_shapes = x
        batch_size = tf.shape(imgs)[0]
        self.resize_ratio = tf.cast(origin_shapes / self.resize_shape,
                                    tf.dtypes.float32)

        self.interpreter.set_tensor(self.input_details[0]['index'], imgs)
        self.interpreter.invoke()
        pred_branches = {}
        for key, output_detail in zip(self.map_keys, self.output_details):
            index = output_detail["index"]
            pred_branches[key] = self.interpreter.get_tensor(index)
            #TODO: get map keys for outputs
        multi_lv_feats = []
        for lv_keys in self.multi_lvs_keys:
            temp = []
            for key in lv_keys:
                temp.append(pred_branches[key])
            multi_lv_feats.append(temp)
        box_results, b_lnmk_outputs = self._anchor_assign(
            batch_size, multi_lv_feats)
        return box_results, b_lnmk_outputs

    # @tf.function
    def _anchor_assign(self, batch_size, multi_lv_feats):
        b_outputs = -tf.ones(shape=(batch_size, self.n_objs,
                                    self.cls_out_channels, 5))
        b_lnmk_outputs = tf.ones(shape=(batch_size, self.n_objs,
                                        self.cls_out_channels, 68, 2)) * np.inf
        obj_start_idx = 0
        b_idx_list, b_bbox_list, b_lnmk_list = [], [], []
        b_kpss_list = []
        for i, (lv_feats,
                stride) in enumerate(zip(multi_lv_feats,
                                         self._feat_stride_fpn)):
            b_cls_preds, reg_feat, x = lv_feats
            feat_h, feat_w = tf.shape(x)[1], tf.shape(x)[2]
            # b_cls_preds = tf.math.sigmoid(b_cls_preds)
            b_mask = b_cls_preds > self.kp_thres
            feat_map_idxs = tf.cast(tf.where(b_mask == True), tf.int32)
            btach_idxs = feat_map_idxs[:, :1]
            b_cls_preds = tf.reshape(b_cls_preds, [-1, self.cls_out_channels])
            mask = b_cls_preds > self.kp_thres
            idxs = tf.where(mask == True)
            channel_idxs = tf.cast(idxs, tf.int32)[:, -1:]
            b_cls_preds = tf.reshape(tf.gather_nd(b_cls_preds, idxs), [-1, 1])
            b_bbox_preds, b_param_preds, b_kp_preds = self._post_convolution(
                i, feat_map_idxs, reg_feat, x)
            b_param_preds = tf.reshape(
                b_param_preds,
                [-1, tf.shape(b_param_preds)[-1] // self._num_anchors])
            b_kp_preds = tf.reshape(
                b_kp_preds,
                [-1, tf.shape(b_kp_preds)[-1] // self._num_anchors])
            b_kp_preds *= stride
            b_bbox_preds = tf.reshape(b_bbox_preds, [-1, 4])
            b_bbox_preds = b_bbox_preds * stride

            height = self.resize_shape[0] // stride
            width = self.resize_shape[1] // stride
            X, Y = tf.meshgrid(tf.range(0, width), tf.range(0, height))
            anchor_centers = tf.stack([X, Y], axis=-1)
            # anchor_centers = tf.reshape(anchor_centers, [-1])
            anchor_centers = tf.reshape((anchor_centers * stride), (-1, 2))
            anchor_centers = tf.reshape(anchor_centers, [feat_h, feat_w, 2])
            anchor_centers = tf.tile(anchor_centers[None, :, :, :],
                                     [batch_size, 1, 1, 1])
            anchor_centers = tf.gather_nd(anchor_centers, feat_map_idxs[:, :3])
            if self._num_anchors > 1:
                anchor_centers = tf.reshape(
                    tf.stack([anchor_centers] * self._num_anchors, axis=1),
                    (-1, 2))
            b_bboxes = self.distance2bbox(anchor_centers, b_bbox_preds)
            b_kp_preds = self.distance2kps(anchor_centers, b_kp_preds)
            N = tf.shape(b_bboxes)[0]
            b_bboxes = tf.reshape(b_bboxes, (N // 2, 2, 4))
            s_idxs = tf.concat(
                [tf.range(0, N // 2, dtype=tf.int32)[:, None], channel_idxs],
                axis=-1)
            b_bboxes = tf.gather_nd(b_bboxes, s_idxs)
            b_param_preds = tf.reshape(
                b_param_preds, (N // 2, 2, tf.shape(b_param_preds)[-1]))

            b_param_preds = tf.gather_nd(b_param_preds, s_idxs)
            b_kp_preds = tf.reshape(b_kp_preds,
                                    (N // 2, 2, tf.shape(b_kp_preds)[-1]))
            b_kp_preds = tf.gather_nd(b_kp_preds, s_idxs)

            b_param_preds = b_param_preds * self.pms[1] + self.pms[0]

            # b_params_list.append(b_param_preds)
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
            b_kp_preds = tf.concat(
                [b_kp_preds[:, :1] + 0.1, b_kp_preds[:, 1:] + 0.5], axis=-1)
            b_kp_preds = tf.einsum('n d, b d -> n d', b_kp_preds[:, ::-1],
                                   self.resize_ratio)
            b_kpss_list.append(b_kp_preds)
            b_outputs = tf.tensor_scatter_nd_update(b_outputs, idxs, b_bboxes)
        xxxx
        b_idx_list = tf.concat(b_idx_list, axis=0)
        b_bbox_list = tf.concat(b_bbox_list, axis=0)
        b_lnmk_list = tf.concat(b_lnmk_list, axis=0)
        # b_params_list = tf.concat(b_params_list, axis=0)
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
        # b_params_list = tf.gather_nd(b_params_list, search_idxs[:, -1:])
        b_lnmk_tls, b_lnmk_brs = tf.math.reduce_min(
            b_lnmk_list, axis=-2), tf.math.reduce_max(b_lnmk_list, axis=-2)
        b_bbox_wh = b_bbox_brs - b_bbox_tls
        b_lnmk_wh = b_lnmk_brs - b_lnmk_tls
        b_scales = b_bbox_wh / b_lnmk_wh
        b_lnmk_list = tf.einsum('n c d, n d -> n c d', b_lnmk_list, b_scales)
        b_lnmk_list = b_lnmk_list + b_kpss_list[:, None, :]
        b_lnmk_outputs = tf.tensor_scatter_nd_update(b_lnmk_outputs,
                                                     b_idx_list, b_lnmk_list)

        b_bboxes = tf.concat(
            [box_results, nms_reuslt[1][..., None], nms_reuslt[2][..., None]],
            axis=-1)
        b_bboxes = tf.where(b_bboxes == -1., np.inf, b_bboxes)
        b_bboxes = tf.reshape(b_bboxes, [-1, self.n_objs, 6])
        return b_bboxes, b_lnmk_outputs

    def _post_convolution(self, stride_i, feat_map_idxs, reg_feat, x):

        def conv(input_data, conv_weights, bias):
            conv_data = input_data * conv_weights
            conv_data = tf.math.reduce_sum(conv_data, axis=[1, 2, 3
                                                            ]) + bias[None, :]
            return conv_data

        padding = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
        reg_feat = tf.pad(reg_feat, padding, "CONSTANT")
        x = tf.pad(x, paddings=padding, mode="CONSTANT")
        N = tf.shape(feat_map_idxs)[0]
        b_idxs = feat_map_idxs[:, :1]
        loc_idxs = feat_map_idxs[:, 1:3] + 1
        bbox_scale, bbox_weight, bbox_bias = self.bbox_scales[
            stride_i], self.bbox_weights[stride_i], self.bbox_biass[stride_i]
        b_grid_kps = loc_idxs[:, None, None, :] + self.grid_3x3[None, :, :, :]
        b_grid_kps = tf.reshape(b_grid_kps, (-1, 2))
        b_grid_kps = tf.concat([tf.tile(b_idxs, [9, 1]), b_grid_kps], axis=-1)
        # -------------------------------------------------- bbox branch
        reg_feat = tf.gather_nd(reg_feat, b_grid_kps)
        reg_feat = tf.reshape(reg_feat, [N, 3, 3, tf.shape(reg_feat)[-1]])
        reg_feat = reg_feat[..., None]
        bbox_weight = tf.tile(bbox_weight[None, ...], [N, 1, 1, 1, 1])

        b_bbox_preds = reg_feat * bbox_weight

        b_bbox_preds = tf.math.reduce_sum(b_bbox_preds,
                                          axis=[1, 2, 3]) + bbox_bias[None, :]
        b_bbox_preds = b_bbox_preds * bbox_scale
        # -------------------------------------------------- params branch
        param_weighs_conv3x3, param_biass_conv3x3, = self.param_weighs_conv3x3[
            2 * stride_i:2 *
            (stride_i + 1)], self.param_biass_conv3x3[2 * stride_i:2 *
                                                      (stride_i + 1)]
        param_weights_pred, param_biass_pred = self.param_weights_pred[
            stride_i], self.param_biass_pred[stride_i]
        b_grid_params = loc_idxs[:, None,
                                 None, :] + self.grid_5x5[None, :, :, :]
        b_grid_params = tf.reshape(b_grid_params, (-1, 2))
        b_grid_params = tf.concat([tf.tile(b_idxs, [25, 1]), b_grid_params],
                                  axis=-1)
        x = tf.gather_nd(x, b_grid_params)
        x = tf.reshape(x, [N, 5, 5, tf.shape(x)[-1]])
        x = x[..., None]
        param_weighs_conv3x3_0 = tf.tile(param_weighs_conv3x3[0][None, ...],
                                         [N, 1, 1, 1, 1])

        param_biass_conv3x3_0 = param_biass_conv3x3[0]

        param_weighs_conv3x3_1 = tf.tile(param_weighs_conv3x3[1][None, ...],
                                         [N, 1, 1, 1, 1])
        param_biass_conv3x3_1 = param_biass_conv3x3[1]
        tmp_x = []

        for i in range(3):
            for j in range(3):
                tmp_x.append(
                    conv(x[:, i:i + 3, j:j + 3], param_weighs_conv3x3_0,
                         param_biass_conv3x3_0)[:, None, :])
        param_x = tf.concat(tmp_x, axis=-2)

        param_x = tf.where(param_x > 0., param_x, 0.)
        param_x = tf.reshape(param_x, (N, 3, 3, tf.shape(param_x)[-1], 1))

        param_x *= param_weighs_conv3x3_1
        param_x = tf.math.reduce_sum(
            param_x, axis=[1, 2, 3]) + param_biass_conv3x3_1[None, :]
        param_x = tf.where(param_x > 0., param_x, 0.)

        param_weights_pred = tf.squeeze(param_weights_pred, axis=0)

        param_x = param_x[:, :, None]
        b_param_preds = tf.math.reduce_sum(param_x * param_weights_pred,
                                           axis=-2) + param_biass_pred[None, :]

        # -------------------------------------------------- kps branch
        kp_weighs_conv3x3, kp_biass_conv3x3, = self.kp_weighs_conv3x3[
            2 * stride_i:2 *
            (stride_i + 1)], self.kp_biass_conv3x3[2 * stride_i:2 *
                                                   (stride_i + 1)]

        kp_weights_pred, kp_biass_pred = self.kp_weights_pred[
            stride_i], self.kp_biass_pred[stride_i]
        kp_weighs_conv3x3_0 = tf.tile(kp_weighs_conv3x3[0][None, ...],
                                      [N, 1, 1, 1, 1])
        kp_biass_conv3x3_0 = kp_biass_conv3x3[0]

        kp_weighs_conv3x3_1 = tf.tile(kp_weighs_conv3x3[1][None, ...],
                                      [N, 1, 1, 1, 1])
        kp_biass_conv3x3_1 = kp_biass_conv3x3[1]

        tmp_x = []
        for i in range(3):
            for j in range(3):
                tmp_x.append(
                    conv(x[:, i:i + 3, j:j + 3], kp_weighs_conv3x3_0,
                         kp_biass_conv3x3_0)[:, None, :])
        kp_x = tf.concat(tmp_x, axis=-2)
        kp_x = tf.where(kp_x > 0., kp_x, 0.)
        kp_x = tf.reshape(kp_x, (N, 3, 3, tf.shape(kp_x)[-1], 1))

        kp_x *= kp_weighs_conv3x3_1
        kp_x = tf.math.reduce_sum(kp_x, axis=[1, 2, 3
                                              ]) + kp_biass_conv3x3_1[None, :]
        kp_x = tf.where(kp_x > 0., kp_x, 0.)

        kp_weights_pred = tf.squeeze(kp_weights_pred, axis=0)

        kp_x = kp_x[:, :, None]
        b_kp_preds = tf.math.reduce_sum(kp_x * kp_weights_pred,
                                        axis=-2) + kp_biass_pred[None, :]
        return b_bbox_preds, b_param_preds, b_kp_preds

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

    # @tf.function
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

    @tf.function
    def angle2matrix(self, angles):
        ''' get rotation matrix from three rotation angles(degree). right-handed.
        Args:
            angles: [3,]. x, y, z angles
            x: pitch. positive for looking down.
            y: yaw. positive for looking left. 
            z: roll. positive for tilting head right. 
        Returns:
            R: [3, 3]. rotation matrix.
        '''
        # use 1 rad =  57.3
        n, _ = angles.get_shape().as_list()
        x, y, z = angles[..., 0], angles[..., 1], angles[..., 2]
        # x, 3, 3
        # for Rx
        row1 = tf.constant([1., 0., 0.], shape=(1, 3))
        row1 = tf.tile(row1[None, :, :], (n, 1, 1))
        row2 = tf.concat([
            tf.zeros(shape=(n, 1, 1)),
            tf.math.cos(x)[..., None, None], -tf.math.sin(x)[..., None, None]
        ],
                         axis=-1)
        row3 = tf.concat([
            tf.zeros(shape=(n, 1, 1)),
            tf.math.sin(x)[..., None, None],
            tf.math.cos(x)[..., None, None]
        ],
                         axis=-1)
        Rx = tf.concat([row1, row2, row3], axis=-2)

        # for Ry
        # y
        row1 = tf.concat([
            tf.math.cos(y)[..., None, None],
            tf.zeros(shape=(n, 1, 1)),
            tf.math.sin(y)[..., None, None]
        ],
                         axis=-1)
        row2 = tf.constant([0., 1., 0.], shape=(1, 3))
        row2 = tf.tile(row2[None, :, :], (n, 1, 1))

        row3 = tf.concat([
            -tf.math.sin(y)[..., None, None],
            tf.zeros(shape=(n, 1, 1)),
            tf.math.cos(y)[..., None, None]
        ],
                         axis=-1)
        Ry = tf.concat([row1, row2, row3], axis=-2)

        # z
        row1 = tf.concat([
            tf.math.cos(z)[..., None, None], -tf.math.sin(z)[..., None, None],
            tf.zeros(shape=(n, 1, 1))
        ],
                         axis=-1)
        row2 = tf.concat([
            tf.math.sin(z)[..., None, None],
            tf.math.cos(z)[..., None, None],
            tf.zeros(shape=(n, 1, 1))
        ],
                         axis=-1)
        row3 = tf.constant([0., 0., 1.], shape=(1, 3))
        row3 = tf.tile(row3[None, :, :], (n, 1, 1))
        Rz = tf.concat([row1, row2, row3], axis=-2)
        R = tf.linalg.matmul(Rz, tf.linalg.matmul(Ry, Rx))
        return R
