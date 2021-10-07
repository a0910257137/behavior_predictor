import numpy as np
import tensorflow as tf
import cv2


class Base:
    def top_k_loc(self, hms, k, h, w, c):
        flat_hms = tf.reshape(hms, [-1, h * w, c])
        flat_hms = tf.transpose(flat_hms, [0, 2, 1])
        scores, indices = tf.math.top_k(flat_hms, k)
        xs = tf.expand_dims(indices % w, axis=-1)
        ys = tf.expand_dims(indices // w, axis=-1)
        b_coors = tf.concat([ys, xs], axis=-1)
        return b_coors

    def apply_max_pool(self, data_in):
        kp_peak = tf.nn.max_pool(input=data_in,
                                 ksize=3,
                                 strides=1,
                                 padding='SAME',
                                 name='hm_nms')
        kps_mask = tf.cast(tf.equal(data_in, kp_peak), tf.float32)
        kps = data_in * kps_mask
        return kps

    def resize_back(self, b_bboxes, resize_ratio):
        """
            Input: b_bboxes shape=[B, C, N, D], B is batch, C is category, N is top-k and  D is (tl, br) y x dimensions
        """
        b, c, n = tf.shape(b_bboxes)[0], tf.shape(b_bboxes)[1], tf.shape(
            b_bboxes)[2]
        b_bboxes = tf.reshape(b_bboxes, [b, c, n, 2, 2])
        b_bboxes = tf.einsum('b c n z d , b d -> b n c z d', b_bboxes,
                             resize_ratio)
        return tf.reshape(b_bboxes, [b, n, c, 4])

    def _object_detector(self, batch_size, hms, size_maps, resize_shape,
                         n_objs, top_k_n, kp_thres):

        hms = self.apply_max_pool(hms)
        mask = hms > kp_thres
        b, h, w, c = tf.shape(hms)[0], tf.shape(hms)[1], tf.shape(
            hms)[2], tf.shape(hms)[3]
        output = -tf.ones(shape=(batch_size, n_objs, c, 5))
        b_coors = self.top_k_loc(hms, top_k_n, h, w, c)
        b_idxs = tf.tile(
            tf.range(0, b, dtype=tf.int32)[:, tf.newaxis, tf.newaxis,
                                           tf.newaxis],
            [1, c, top_k_n, 1],
        )
        b_infos = tf.concat([b_idxs, b_coors], axis=-1)

        b_size_vals = tf.gather_nd(size_maps, b_infos)

        b_c_idxs = tf.tile(
            tf.range(0, c, dtype=tf.int32)[tf.newaxis, :, tf.newaxis,
                                           tf.newaxis], [b, 1, top_k_n, 1])

        b_infos = tf.concat([b_infos, b_c_idxs], axis=-1)
        b_scores = tf.gather_nd(hms, b_infos)

        b_centers = tf.cast(b_coors, tf.float32)

        b_tls = (b_centers - b_size_vals / 2)
        b_brs = (b_centers + b_size_vals / 2)
        # clip value
        b_br_y = b_brs[..., 0]
        b_br_x = b_brs[..., 1]
        b_tls = tf.where(b_tls < 0., 0., b_tls)

        b_br_y = tf.where(b_brs[..., :1] > resize_shape[0] - 1.,
                          resize_shape[0] - 1., b_brs[..., :1])
        b_br_x = tf.where(b_brs[..., -1:] > resize_shape[1] - 1.,
                          resize_shape[1] - 1., b_brs[..., -1:])
        b_brs = tf.concat([b_br_y, b_br_x], axis=-1)

        b_bboxes = tf.concat([b_tls, b_brs], axis=-1)
        b_bboxes = self.resize_back(b_bboxes, resize_ratio)

        b_scores = tf.transpose(b_scores, [0, 2, 1])

        b_bboxes = tf.concat([b_bboxes, b_scores[..., None]], axis=-1)

        mask = b_scores > kp_thres
        index = tf.where(mask == True)
        n = tf.shape(index)[0]
        d = tf.shape(b_bboxes)[-1]
        c_idx = tf.tile(tf.range(d)[None, :, None], [n, 1, 1])
        index = tf.cast(tf.tile(index[:, tf.newaxis, :], [1, d, 1]), tf.int32)
        index = tf.concat([index, c_idx], axis=-1)
        output = tf.tensor_scatter_nd_update(output, index, b_bboxes[mask])
        scores = output[..., -1]
        output = output[..., :-1]
        nms_reuslt = tf.image.combined_non_max_suppression(
            output,
            scores,
            n_objs,
            n_objs,
            iou_threshold=self.nms_iou_thres,
            clip_boxes=False)
        b_bboxes = tf.concat([
            nms_reuslt[0], nms_reuslt[1][..., None], nms_reuslt[2][..., None]
        ],
                             axis=-1)
        b_bboxes = tf.where(b_bboxes == -1., np.inf, b_bboxes)
        b_bboxes = tf.reshape(b_bboxes, [-1, n_objs, 6])
        return b_bboxes
