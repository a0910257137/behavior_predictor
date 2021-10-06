from pprint import pprint
from .base import Base
import numpy as np
import tensorflow as tf


class PostModel(tf.keras.Model):
    def __init__(self, pred_model, n_objs, k_pairings, top_k_n, kp_thres,
                 nms_iou_thres, resize_shape, *args, **kwargs):
        super(PostModel, self).__init__(*args, **kwargs)
        self.pred_model = pred_model
        self.n_objs = n_objs
        self.top_k_n = top_k_n
        self.kp_thres = kp_thres
        self.nms_iou_thres = nms_iou_thres
        self.resize_shape = tf.cast(resize_shape, tf.float32)
        self.k_pairings = k_pairings
        self.base = Base()

    @tf.function
    def call(self, x, training=False):
        imgs, origin_shapes = x
        batch_size = tf.shape(imgs)[0]
        self.resize_ratio = tf.cast(origin_shapes / self.resize_shape,
                                    tf.dtypes.float32)
        preds = self.pred_model(imgs, training=False)
        b_bboxes = self._obj_detect(batch_size, preds['obj_heat_map'],
                                    preds['obj_size_maps'])
        b_relations = self._relations(batch_size, b_bboxes, preds['rel_maps'],
                                      preds['embed_maps'])

        return b_bboxes, b_relations

    @tf.function
    def _relations(self, batch_size, b_bboxes, b_rel_maps, b_embed_maps):
        b_center_kps = tf.einsum('b n d ,b d ->  b n d',
                                 (b_bboxes[..., :2] + b_bboxes[..., 2:4]) / 2,
                                 1 / self.resize_ratio)

        b_center_kps = tf.where(b_center_kps == np.inf, 0., b_center_kps)
        b_idxs = tf.range(batch_size, dtype=tf.int32)
        b_idx_infos = tf.tile(b_idxs[:, None, None], [1, self.n_objs, 1])
        b_center_kps = tf.cast((b_center_kps + .5), tf.int32)
        b_idx_infos = tf.concat([b_idx_infos, b_center_kps], axis=-1)
        b_rel_vals = tf.gather_nd(b_rel_maps, b_idx_infos)
        b_rel_vals = tf.nn.softmax(b_rel_vals, axis=-1)
        b_embed_vals = tf.gather_nd(b_embed_maps, b_idx_infos)
        embedding_chs = tf.shape(b_embed_vals)[-1]
        b_sub = b_embed_vals[..., :embedding_chs // 2][0]

        b_obj = b_embed_vals[..., embedding_chs // 2:][0]

        b_subs = tf.expand_dims(b_embed_vals[..., :embedding_chs // 2],
                                axis=-1)
        b_objs = tf.expand_dims(b_embed_vals[..., embedding_chs // 2:],
                                axis=-1)
        b_subs = tf.transpose(b_subs, [0, 3, 2, 1])

        b_diff_vals = b_subs - b_objs
        b_diff_vals = tf.sqrt(
            tf.math.reduce_sum(tf.square(b_diff_vals), axis=-2))

        b_mask = tf.where(b_center_kps == 0, 0., 1.)
        b, n, c = [tf.shape(b_rel_vals)[i] for i in range(3)]
        b_rel_mask = b_mask[..., :1]
        b_rel_vals = b_rel_vals * \
            tf.tile(b_mask[..., :1], [1,  1, c])

        b_idxs_mask = tf.tile(b_mask[..., :1], [1, 1, 3])

        b_mask = tf.matmul(b_mask, b_mask, transpose_b=True) / 2
        b_diff_vals = b_diff_vals * b_mask
        b_diff_vals = tf.where(b_diff_vals == 0., np.inf, b_diff_vals)
        b_diff_vals = tf.transpose(b_diff_vals, [0, 2, 1])

        b_objs_idxs = tf.nn.top_k(-b_diff_vals, k=self.k_pairings)
        b_pairing_infos = b_objs_idxs[1]
        b_subs_idxs = tf.tile(tf.range(n, dtype=tf.int32)[None, :], [b, 1])

        b_subs_idxs = tf.expand_dims(b_subs_idxs, axis=-1) + 1
        b_rel_inds = tf.expand_dims(
            tf.argmax(b_rel_vals, output_type=tf.int32, axis=-1), axis=-1) + 1

        b_relations = [
            tf.concat([
                b_subs_idxs, b_rel_inds,
                tf.expand_dims(b_pairing_infos[..., i] + 1, axis=-1)
            ],
                      axis=-1) for i in range(self.k_pairings)
        ]
        b_relations = tf.concat(b_relations, axis=-2)
        b_rel_mask = tf.tile(b_rel_mask, [1, self.k_pairings, 3])

        b_relations = tf.cast(b_relations, tf.float32) * b_rel_mask
        b_relations = tf.where(b_relations == 0., np.inf, b_relations)
        return b_relations - 1.

    @tf.function
    def _obj_detect(self, batch_size, hms, size_maps):
        hms = self.base.apply_max_pool(hms)
        mask = hms > self.kp_thres
        b, h, w, c = tf.shape(hms)[0], tf.shape(hms)[1], tf.shape(
            hms)[2], tf.shape(hms)[3]
        output = -tf.ones(shape=(batch_size, self.n_objs, c, 5))
        b_coors = self.base.top_k_loc(hms, self.top_k_n, h, w, c)
        b_idxs = tf.tile(
            tf.range(0, b, dtype=tf.int32)[:, tf.newaxis, tf.newaxis,
                                           tf.newaxis],
            [1, c, self.top_k_n, 1],
        )
        b_infos = tf.concat([b_idxs, b_coors], axis=-1)

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

        nms_reuslt = tf.image.combined_non_max_suppression(
            output,
            scores,
            self.n_objs,
            self.n_objs,
            iou_threshold=self.nms_iou_thres,
            clip_boxes=False)
        b_bboxes = tf.concat([
            nms_reuslt[0], nms_reuslt[1][..., None], nms_reuslt[2][..., None]
        ],
                             axis=-1)
        b_bboxes = tf.where(b_bboxes == -1., np.inf, b_bboxes)
        b_bboxes = tf.reshape(b_bboxes, [-1, self.n_objs, 6])
        return b_bboxes
