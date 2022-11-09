from pprint import pprint
from .base import Base
from utils.io import load_BFM
import numpy as np
import tensorflow as tf


class TDMMPostModel(tf.keras.Model):

    def __init__(self, tdmm_cfg, pred_model, n_objs, top_k_n, kp_thres,
                 nms_iou_thres, resize_shape, *args, **kwargs):
        super(TDMMPostModel, self).__init__(*args, **kwargs)
        self.n_s, self.n_Rt = tdmm_cfg['n_s'], tdmm_cfg['n_Rt']
        self.n_shp, self.n_exp = tdmm_cfg['n_shp'], tdmm_cfg['n_exp']
        pms = tf.cast(np.load(tdmm_cfg['pms_path']), tf.float32)

        pms_s, pms_R = pms[:, :self.n_s], pms[:, self.n_s:self.n_s + self.n_Rt]
        pms_shp, pms_exp = pms[:, self.n_s + self.n_Rt:self.n_s + self.n_Rt +
                               self.n_shp], pms[:, 211:]
        pms = tf.concat([pms_s, pms_R, pms_shp, pms_exp], axis=-1)
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
        mean = tf.math.reduce_mean(self.u_base, axis=0, keepdims=True)
        self.u_base -= mean
        self.u_base = tf.reshape(self.u_base, (tf.shape(self.u_base)[0] * 3, 1))

        self.shp_base = tf.cast(head_model['shapePC'], tf.float32)[:, :50]
        self.shp_base = tf.gather(self.shp_base, self.valid_ind)
        self.exp_base = tf.cast(head_model['expPC'], tf.float32)
        self.exp_base = tf.gather(self.exp_base, self.valid_ind)

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
        box_results, output_lnmks, output_pose = self._reconstruct_tdmm(
            batch_size, preds["obj_heat_map"], preds['obj_param_map'])
        return box_results, output_lnmks, output_pose

    # @tf.function
    def _reconstruct_tdmm(self, batch_size, hms, pms):
        hms = self.base.apply_max_pool(hms)
        b, h, w, c = [tf.shape(hms)[i] for i in range(4)]
        output_bboxes = -tf.ones(shape=(batch_size, c, self.n_objs, 5))
        b_coors = self.base.top_k_loc(hms, self.top_k_n, h, w, c)

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
        b_params = b_params * self.pms[1] + self.pms[0]
        b_s, b_Rt = b_params[..., :self.n_s], b_params[..., self.n_s:self.n_s +
                                                       self.n_Rt]

        b_shp = b_params[...,
                         self.n_s + self.n_Rt:self.n_s + self.n_Rt + self.n_shp]
        b_exp = b_params[..., self.n_s + self.n_Rt + self.n_shp:]

        b_coors = tf.reshape(b_coors[b_mask],
                             (batch_size, -1, tf.shape(b_coors)[-1]))
        n = tf.shape(b_coors)[-2]
        vertices = self.u_base + tf.linalg.matmul(
            self.shp_base, b_shp[..., None]) + tf.linalg.matmul(
                self.exp_base, b_exp[..., None])

        vertices = tf.reshape(vertices,
                              (batch_size, n, tf.shape(vertices)[-2] // 3, 3))
        b_Rt = tf.concat(
            [b_Rt, tf.ones(shape=(batch_size, tf.shape(b_Rt)[1], 1))], axis=-1)

        b_Rt = tf.reshape(b_Rt, [batch_size, n, 3, 4])
        b_R = b_Rt[..., :-1]
        b_t = b_Rt[..., -1]
        b_t = b_t[:, :, None, :]
        b_lnmks = b_s[..., tf.newaxis] * tf.linalg.matmul(
            vertices, b_R, transpose_b=(0, 1, 3, 2))
        # t_vertices = tf.transpose(tf.reshape(t_vertices, (batch_size, n, -1)),
        #                           [2, 0, 1])
        # b_lnmks = tf.transpose(tf.gather(t_vertices, self.valid_ind), (1, 2, 0))
        # b_lnmks = tf.reshape(b_lnmks,
        #                      (batch_size, n, tf.shape(b_lnmks)[-1] // 3, 3))
        # b_centroid = tf.math.reduce_mean(b_lnmks, axis=-2, keepdims=True)
        # b_lnmks = b_lnmks + b_t
        b_coors = tf.cast(b_coors, tf.float32)

        b_coors = tf.einsum('b n c, b c -> b n c', b_coors, self.resize_ratio)
        b_coors = b_coors[..., ::-1]
        b_lnmks = b_lnmks[..., :2] + b_coors[:, :, None, :]
        b_lnmks = b_lnmks[..., :2][..., ::-1]
        # b_lnmks = tf.einsum('b n k c, b c -> b n k c', b_lnmks,
        #                     self.resize_ratio)
        b_tls = tf.math.reduce_min(b_lnmks, axis=2)
        b_brs = tf.math.reduce_max(b_lnmks, axis=2)
        b_scores = tf.reshape(b_scores[b_mask],
                              (batch_size, -1, tf.shape(b_scores)[-1]))

        b_bboxes = tf.concat([b_tls, b_brs, b_scores], axis=-1)
        b_bboxes = tf.reshape(b_bboxes, [-1, 5])
        index = tf.where(b_mask == True)
        # [0, 0, 13]

        d = tf.shape(b_bboxes)[-1]
        c_idx = tf.tile(tf.range(d)[None, :, None], [n, 1, 1])
        index = tf.cast(tf.tile(index[:, tf.newaxis, :], [1, d, 1]), tf.int32)
        index = tf.concat([index, c_idx], axis=-1)
        output_bboxes = tf.tensor_scatter_nd_update(output_bboxes, index,
                                                    b_bboxes)
        output_bboxes = tf.transpose(output_bboxes, [0, 2, 1, 3])
        scores = output_bboxes[..., -1]
        output_bboxes = output_bboxes[..., :-1]
        # [B, N, Cate, 4]
        nms_reuslt = tf.image.combined_non_max_suppression(
            output_bboxes,
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
        output_lnmks, output_pose = self._valid_lnmks(batch_size, b_bboxes,
                                                      b_scores, b_lnmks, b_R)
        return b_bboxes, output_lnmks, output_pose

    def _valid_lnmks(self, batch_size, b_bboxes, b_scores, b_lnmks, b_R):
        '''
        Args:
            batch_size
            b_bboxes: (batch_size, N, 6) y1, x1, y2, x2, score, cate
            b_scores: (batch_size, N, C) heatmap scores
            b_lnmks: (batch_size, N, C) heatmap scores
        Returns:
            landmarks (batch_size, N, 68, 2)
            R : (pitch, yaw, roll)
        '''
        _, _, k, c = [tf.shape(b_lnmks)[i] for i in range(4)]
        output_lnmks = -tf.ones(shape=(batch_size, self.n_objs, k * c))
        output_pose = -tf.ones(shape=(batch_size, self.n_objs, 3))
        b_mask = tf.math.reduce_all(b_bboxes != np.inf, axis=-1)
        b_valid_bboxes = tf.reshape(b_bboxes[b_mask], (batch_size, -1, 6))
        b_valid_scores = b_valid_bboxes[..., -2:-1]
        T = b_scores - tf.transpose(b_valid_scores, (0, 2, 1))
        b_idxs = tf.where(T == 0.)[..., :2]
        b_lnmks = tf.gather_nd(b_lnmks, b_idxs)
        b_lnmks = tf.reshape(b_lnmks, (-1, k * c))

        n = tf.shape(b_lnmks)[0] // batch_size
        c_idx = tf.tile(tf.range(k * c)[None, :, None], [n, 1, 1])
        index = tf.cast(tf.tile(b_idxs[:, tf.newaxis, :], [1, k * c, 1]),
                        tf.int32)
        index = tf.concat([index, c_idx], axis=-1)
        output_lnmks = tf.tensor_scatter_nd_update(output_lnmks, index, b_lnmks)
        output_lnmks = tf.reshape(output_lnmks, (batch_size, self.n_objs, k, c))
        b_R = tf.gather_nd(b_R, b_idxs)
        b_yaw = tf.math.asin(-b_R[..., 2, 0]) * (180 / np.pi)
        b_pitch = tf.math.atan2(
            b_R[..., 2, 1] / tf.math.cos(b_yaw),
            b_R[..., 2, 2] / tf.math.cos(b_yaw)) * (180 / np.pi)
        b_roll = tf.math.atan2(
            b_R[..., 1, 0] / tf.math.cos(b_yaw),
            b_R[..., 0, 0] / tf.math.cos(b_yaw)) * (180 / np.pi)
        b_pose = tf.concat(
            [b_pitch[..., None], b_yaw[..., None], b_roll[..., None]], axis=-1)
        c_idx = tf.tile(tf.range(3)[None, :, None], [n, 1, 1])
        index = tf.cast(tf.tile(b_idxs[:, tf.newaxis, :], [1, 3, 1]), tf.int32)
        index = tf.concat([index, c_idx], axis=-1)
        output_pose = tf.tensor_scatter_nd_update(output_pose, index, b_pose)
        output_lnmks = tf.where(output_lnmks == -1., np.inf, output_lnmks)
        output_pose = tf.where(output_pose == -1., np.inf, output_pose)
        return output_lnmks, output_pose
