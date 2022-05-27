import numpy as np
import tensorflow as tf
from .base import Base
from pprint import pprint
from glob import glob
import os


class Optimize:
    def __init__(self, interpreter, weight_root, n_objs, top_k_n, kp_thres,
                 nms_iou_thres, resize_shape, *args, **kwargs):
        self.interpreter = interpreter
        self.interpreter.allocate_tensors()
        self.n_objs = n_objs
        self.top_k_n = top_k_n
        self.kp_thres = kp_thres
        self.nms_iou_thres = nms_iou_thres
        self.resize_shape = tf.cast(resize_shape, tf.float32)

        reconv_dir = os.path.join(weight_root, "experiment")
        sm_dir = os.path.join(weight_root, "size")
        om_dir = os.path.join(weight_root, "offset")

        self.reconvs_dict = self._load_weights(
            glob(os.path.join(reconv_dir, '*.npy')))
        self.sms_dict = self._load_weights(glob(os.path.join(sm_dir, '*.npy')))
        self.oms_dict = self._load_weights(glob(os.path.join(om_dir, '*.npy')))
        self.map_keys = ["hms", "x"]
        self.grid = [
            [[-1, -1], [-1, 0], [-1, 1]],
            [[0, -1], [0, 0], [0, 1]],
            [[1, -1], [1, 0], [1, 1]],
        ]
        self.grid = tf.cast(np.asarray(self.grid), tf.int32)
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_is_FP = self.input_details[0]['dtype'] == np.float32
        self.output_is_FP = False

        for d in self.output_details:
            if d['dtype'] == np.float32:
                self.output_is_FP = True
        self.base = Base()

    def __call__(self, x, training=False):
        imgs, origin_shapes = x
        batch_size = tf.shape(imgs)[0]
        self.resize_ratio = tf.cast(origin_shapes / self.resize_shape,
                                    tf.dtypes.float32)
        if not self.input_is_FP:
            imgs = self._quantized(imgs, self.input_details)

        self.interpreter.set_tensor(self.input_details[0]['index'], imgs)
        self.interpreter.invoke()
        pred_branches = {}
        # the magix conv 1x1 should have odd number
        for key, output_detail in zip(self.map_keys, self.output_details):
            pred_maps = self.interpreter.get_tensor(output_detail['index'])
            if not self.output_is_FP:
                pred_maps = self._dequantized(pred_maps, output_detail)
            if key == 'hms':
                pred_branches[key] = tf.cast(pred_maps[..., :2], tf.float32)
            else:
                pred_branches[key] = tf.cast(pred_maps, tf.float32)

        b_bboxes, b_lnmks, b_nose_scores = self._obj_detect(
            batch_size, pred_branches["hms"], pred_branches['x'])
        return b_bboxes, b_lnmks, b_nose_scores

    def _quantized(self, map_vals, infos):
        scale, zeros_point = infos[0]['quantization']
        q_map_vals = ((1 / scale) * map_vals - zeros_point)

        q_map_vals = tf.cast((q_map_vals + 0.01), tf.uint8)
        return q_map_vals

    def _dequantized(self, map_vals, infos):
        scale, zeros_point = infos['quantization']
        shifted_vals = map_vals.astype(
            np.float32) - np.asarray(zeros_point).astype(np.float32)
        de_map_vals = np.asarray(scale).astype(np.float32) * shifted_vals
        return de_map_vals

    @tf.function
    def _obj_detect(self, batch_size, hms, x):
        hms = self.base.apply_max_pool(hms)
        b, h, w, c = [tf.shape(hms)[i] for i in range(4)]
        b_coors = self.base.top_k_loc(hms, self.top_k_n, h, w, c)
        b_lnmks = b_coors[:, 1:, ...]
        b_coors = b_coors[:, :1, ...]

        res_c = c - 1
        c = c - res_c
        output = -tf.ones(shape=(batch_size, self.n_objs, c, 5))
        b_idxs = tf.tile(
            tf.range(0, b, dtype=tf.int32)[:, tf.newaxis, tf.newaxis,
                                           tf.newaxis],
            [1, c, self.top_k_n, 1],
        )
        #offset vectors
        b_lnmk_point_vectors, b_grid_kps, b_lnmks, b_lnmk_scores, _ = self._point_vectors(
            batch_size, b_idxs, x, hms[..., 1], b_lnmks)
        b_conv_x = self._reconv(b_lnmk_point_vectors)
        b_conv_x = self._seperatable(batch_size, b_conv_x, b_grid_kps,
                                     self.oms_dict)
        b_offsets = self._project_preds(b_conv_x, self.oms_dict)

        #size map vectors
        b_coors_point_vectors, b_grid_kps, b_kps, b_coor_scores, mask = self._point_vectors(
            batch_size, b_idxs, x, hms[..., 0], b_coors)
        b_conv_x = self._reconv(b_coors_point_vectors)

        b_conv_x = self._seperatable(batch_size, b_conv_x, b_grid_kps,
                                     self.sms_dict)
        b_sizes = self._project_preds(b_conv_x, self.sms_dict)
        b_kps = tf.cast(b_kps, tf.float32)
        b_tls = b_kps - b_sizes / 2
        b_brs = b_kps + b_sizes / 2
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

        b_bboxes = b_bboxes[:, None, :]
        b_bboxes = self.base.resize_back(b_bboxes, self.resize_ratio)
        # B N C D
        b_bboxes = tf.concat([b_bboxes, b_coor_scores[..., None, None]],
                             axis=-1)

        n = tf.shape(b_bboxes)[1]
        d = tf.shape(b_bboxes)[-1]
        index = tf.where(mask == True)

        index = tf.concat([index[..., :1], index[..., -1:], index[..., 1:2]],
                          axis=-1)
        c_idx = tf.tile(tf.range(d)[None, :, None], [n, 1, 1])
        index = tf.cast(tf.tile(index[:, tf.newaxis, :], [1, d, 1]), tf.int32)
        index = tf.concat([index, c_idx], axis=-1)
        output = tf.tensor_scatter_nd_update(output, index,
                                             tf.reshape(b_bboxes, [n, 5]))
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
        # b_lnmks, b_lnmk_scores
        b_lnmks = self._offset_vec_nose(b_lnmks, b_offsets)
        return b_bboxes, b_lnmks, b_lnmk_scores

    def _reconv(self, b_point_vectors):

        keys = ['conv_1x1_kernel.npy', 'conv_1x1_bias.npy']
        conv_1x1_kernel = self.reconvs_dict['conv_1x1_kernel.npy']
        conv_1x1_bias = self.reconvs_dict['conv_1x1_bias.npy']
        conv_1x1_kernel = tf.reshape(
            conv_1x1_kernel,
            [-1] + [tf.shape(conv_1x1_kernel)[i] for i in range(4)])
        conv_1x1_kernel = conv_1x1_kernel[:, :, :, None, :, :]
        b_point_vectors = b_point_vectors[..., None]
        # convolution 1x1
        b_point_vectors = b_point_vectors * conv_1x1_kernel
        # merge convolution values
        b_point_vectors = tf.math.reduce_sum(b_point_vectors, axis=4)
        b_point_vectors = b_point_vectors + conv_1x1_bias
        # relu activation
        return tf.math.maximum(0.0, b_point_vectors)

    def _seperatable(self, batch_size, b_conv_x, b_kps, weight_dict):
        keys = [
            'conv_3x3_depth.npy', 'conv_3x3_point.npy', 'conv_3x3_bias.npy',
            'conv_1x1_kernel.npy', 'conv_1x1_bias.npy'
        ]
        _, _, _, n, d = [tf.shape(b_kps)[i] for i in range(5)]
        self.grid = self.grid[None, :, :, None, :]

        conv_3x3_depth = weight_dict['conv_3x3_depth.npy']
        conv_3x3_point = weight_dict['conv_3x3_point.npy']
        conv_3x3_bias = weight_dict['conv_3x3_bias.npy']

        #-------------------------conv3x3-------------------
        conv_3x3_depth = tf.tile(conv_3x3_depth[None, :, :, None, :, :],
                                 [1, 1, 1, n, 1, 1])
        b_conv_x = b_conv_x[..., None] * conv_3x3_depth
        b_conv_x = tf.math.reduce_sum(b_conv_x, axis=[1, 2], keepdims=True)

        point_channel = conv_3x3_point.shape[-1]
        b_conv_x = tf.tile(b_conv_x, [1, 1, 1, 1, 1, point_channel])

        conv_3x3_point = tf.tile(conv_3x3_point[None, :, :, None, :, :],
                                 [1, 1, 1, n, 1, 1])
        b_conv_x = b_conv_x * conv_3x3_point
        b_conv_x = tf.math.reduce_sum(b_conv_x, axis=4)
        b_conv_x += conv_3x3_bias
        return tf.math.maximum(0.0, b_conv_x)

    def _project_preds(self, b_conv_x, weight_dict):
        _, _, _, n, c = [tf.shape(b_conv_x)[i] for i in range(5)]
        conv_1x1_kernel = weight_dict['conv_1x1_kernel.npy']
        conv_1x1_bias = weight_dict['conv_1x1_bias.npy']
        conv_1x1_kernel = conv_1x1_kernel[None, :, :, None, :, :]
        conv_1x1_kernel = tf.tile(conv_1x1_kernel, [1, 1, 1, n, 1, 1])
        b_conv_x = b_conv_x[..., None] * conv_1x1_kernel
        b_conv_x = tf.math.reduce_sum(b_conv_x, axis=[1, 2, 4])
        b_conv_x += conv_1x1_bias
        return b_conv_x

    def _point_vectors(self, batch_size, b_idxs, x_maps, feat_maps, b_kps):
        def boundary(b_kps, b_scores):
            b_kps_y, b_kps_x = b_kps[:, 0], b_kps[:, 1]

            mask_y = tf.cast(b_kps_y < 191, tf.float32) * tf.cast(
                b_kps_y > 0, tf.float32)

            mask_x = tf.cast(b_kps_x < 319, tf.float32) * tf.cast(
                b_kps_x > 0, tf.float32)
            mask = tf.cast(mask_y * mask_x, tf.bool)
            b_kps = b_kps[mask]
            b_scores = b_scores[mask]
            return b_kps, b_scores

        scores = tf.gather_nd(feat_maps, tf.concat([b_idxs, b_kps], axis=-1))
        mask = scores > 0.5
        b_kps, b_scores = b_kps[mask], scores[mask]
        b_kps, b_scores = boundary(b_kps, b_scores)
        b_grid_kps = b_kps

        n, d = [tf.shape(b_grid_kps)[i] for i in range(2)]
        b_kps = tf.reshape(b_kps, (-1, n, d))
        b_grid_kps = tf.reshape(b_grid_kps, (batch_size, n, d))
        b_scores = tf.reshape(b_scores, (batch_size, n))

        _, n, d = [tf.shape(b_grid_kps)[i] for i in range(3)]
        b_grid_kps = b_grid_kps[:, None, None, :, :]
        self.grid = self.grid[None, :, :, None, :]

        # in order to fit the seperatable convolution
        b_grid_kps = b_grid_kps + self.grid
        # add a check is out side grid excluded it

        grid_n = 9
        b_grid_kps = tf.reshape(b_grid_kps, (1, grid_n * n, d))
        b_idx = tf.tile(
            tf.range(batch_size, dtype=tf.int32)[:, None, None],
            [1, grid_n * n, 1])
        b_idx = tf.concat([b_idx, b_grid_kps], axis=-1)
        b_x_point_vectors = tf.gather_nd(x_maps, b_idx)
        b_x_point_vectors = tf.reshape(
            b_x_point_vectors, (1, 3, 3, n, b_x_point_vectors.shape[-1]))
        b_grid_kps = tf.reshape(b_grid_kps, (1, 3, 3, n, 2))

        return b_x_point_vectors, b_grid_kps, b_kps, b_scores, mask

    def _offset_vec_nose(self, b_lnmks, b_offsets):
        _, n, d = [tf.shape(b_lnmks)[i] for i in range(3)]
        b_offsets = tf.reshape(b_offsets, (-1, n, 4, 2))
        b_lnmks = tf.cast(b_lnmks[:, :, None, :], tf.float32)
        b_ENM = b_lnmks - b_offsets
        b_lnmks = tf.concat([b_ENM[:, :, :2], b_lnmks, b_ENM[:, :, 2:, :]],
                            axis=-2)
        # magic rounding shift-pixels
        b_lnmks += 0.5
        b_lnmks = tf.einsum('b n c d, b d -> b n c d', b_lnmks,
                            self.resize_ratio)
        return b_lnmks

    def _load_weights(self, paths):
        output_dict = {}
        for path in paths:
            name = path.split('/')[-1]
            weights = np.load(path)
            output_dict[name] = tf.cast(weights, tf.float32)
        return output_dict