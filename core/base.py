import tensorflow as tf


class Base:

    def top_k_loc(self, hms, k, h, w, c):
        flat_hms = tf.reshape(hms, [-1, h * w, c])
        flat_hms = tf.transpose(flat_hms, [0, 2, 1])
        scores, indices = tf.math.top_k(flat_hms, k, sorted=False)
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