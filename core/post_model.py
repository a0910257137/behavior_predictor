import tensorflow as tf
import numpy as np
from pprint import pprint


class PostModel(tf.keras.Model):
    def __init__(self, img_input_size, model, strides, scale_factor, reg_max,
                 top_k_n, iou_thres, box_score, *args, **kwargs):
        super(PostModel, self).__init__(*args, **kwargs)
        self.strides = tf.constant(strides, dtype=tf.float32)
        self.scale_factor = scale_factor
        self.reg_max = reg_max
        self.top_k_n = top_k_n
        self.iou_thres = iou_thres
        self.box_score = box_score
        self.model = model
        self.layer_keys = [
            'layer_output_0', 'layer_output_1', 'layer_output_2'
        ]
        self.model_inp_shape = tf.constant(img_input_size, dtype=tf.float32)

    def call(self, inputs, training=None):
        x, origin_shapes = inputs
        self.batch = tf.shape(origin_shapes)[0]
        resize_ratio = tf.einsum('b d, d ->b d', origin_shapes,
                                 1 / self.model_inp_shape)
        ml_intput_shapes = tf.tile(self.model_inp_shape[None, :], [3, 1])
        preds = self.model(x, training)
        # cls_scores, bbox_preds = [], []
        mlvl_bboxes = [[], [], []]
        mlvl_scores = [[], [], []]
        for i, k in enumerate(self.layer_keys):
            for j in range(self.batch):
                cls_score, bbox_pred = preds[k]['cls_scores'][j], preds[k][
                    'bbox_pred'][j]
                bboxes, scores = self.get_bboxes(cls_score, bbox_pred,
                                                 self.strides[i],
                                                 ml_intput_shapes[i],
                                                 resize_ratio)
                mlvl_bboxes[i].append(bboxes)
                mlvl_scores[i].append(scores)
            # mlvl_bboxes.append(bboxes)
            # mlvl_scores.append(scores)
        batch_bboxes = []
        batch_scores = []
        for j in range(self.batch):
            mlvl_bboxes = tf.concat(
                [mlvl_bboxes[0][j], mlvl_bboxes[1][j], mlvl_bboxes[2][j]],
                axis=0)
            mlvl_scores = tf.concat(
                [mlvl_scores[0][j], mlvl_scores[1][j], mlvl_scores[2][j]],
                axis=0)
            # add a dummy background class at the end of all labels
            padding = tf.zeros_like(mlvl_scores[..., :1])
            mlvl_scores = tf.concat([mlvl_scores, padding], axis=-1)
            mlvl_bboxes = mlvl_bboxes[:, None, :]
            mlvl_scores = mlvl_scores[..., :-1]
            batch_bboxes.append(mlvl_bboxes)
            batch_scores.append(mlvl_scores)
        batch_bboxes = tf.stack(batch_bboxes)
        batch_scores = tf.stack(batch_scores)
        nms_reuslt = tf.image.combined_non_max_suppression(batch_bboxes,
                                                           batch_scores,
                                                           self.top_k_n,
                                                           self.top_k_n,
                                                           iou_threshold=0.5,
                                                           clip_boxes=False)
        resize_ratio = tf.tile(resize_ratio, [1, 2])
        b_bboxes = tf.einsum('b n d,b d ->b  n d', nms_reuslt[0], resize_ratio)
        mask = nms_reuslt[1] > self.box_score
        b_bboxes = tf.concat([
            nms_reuslt[0], nms_reuslt[1][..., None], nms_reuslt[2][..., None]
        ],
                             axis=-1)
        rets = tf.reshape(b_bboxes[mask], [self.batch, -1, 6])
        return rets
        # #TODO: in the return layers
        # mlvl_bboxes = tf.concat(mlvl_bboxes, axis=1)
        # mlvl_scores = tf.concat(mlvl_scores, axis=1)

        # # add a dummy background class at the end of all labels
        # padding = tf.zeros_like(mlvl_scores[..., :1])
        # mlvl_scores = tf.concat([mlvl_scores, padding], axis=-1)

        # mlvl_bboxes = mlvl_bboxes[:, :, None, :]
        # mlvl_scores = mlvl_scores[..., :-1]

        # nms_reuslt = tf.image.combined_non_max_suppression(mlvl_bboxes,
        #                                                    mlvl_scores,
        #                                                    self.top_k_n,
        #                                                    self.top_k_n,
        #                                                    iou_threshold=0.5,
        #                                                    clip_boxes=False)
        # resize_ratio = tf.tile(resize_ratio, [1, 2])
        # b_bboxes = tf.einsum('b n d,b d ->b  n d', nms_reuslt[0], resize_ratio)
        # mask = nms_reuslt[1] > self.box_score
        # b_bboxes = tf.concat([
        #     nms_reuslt[0], nms_reuslt[1][..., None], nms_reuslt[2][..., None]
        # ],
        #                      axis=-1)
        # rets = tf.reshape(b_bboxes[mask], [self.batch, -1, 6])
        # return rets

    def get_single_level_center_point(self,
                                      featmap_size,
                                      stride,
                                      flatten=True):
        """
        Generate pixel centers of a single stage feature map.
        :param featmap_size: height and width of the feature map
        :param stride: down sample stride of the feature map
        :param dtype: data type of the tensors
        :param device: device of the tensors
        :param flatten: flatten the x and y tensors
        :return: y and x of the center points
        """
        # 40 , 32
        h, w = featmap_size
        y_range = (np.arange(h) + 0.5) * stride
        x_range = (np.arange(w) + 0.5) * stride
        y, x = tf.meshgrid(y_range, x_range)
        y = tf.transpose(y)
        x = tf.transpose(x)
        if flatten:
            y = np.reshape(y, [-1])
            x = np.reshape(x, [-1])
        return y, x

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
        y1 = points[..., 0] - distance[..., 0]
        x1 = points[..., 1] - distance[..., 1]
        y2 = points[..., 0] + distance[..., 2]
        x2 = points[..., 1] + distance[..., 3]
        if max_shape is not None:
            y1 = tf.clip_by_value(y1,
                                  clip_value_min=0.,
                                  clip_value_max=max_shape[0])
            x1 = tf.clip_by_value(x1,
                                  clip_value_min=0.,
                                  clip_value_max=max_shape[1])
            y2 = tf.clip_by_value(y2,
                                  clip_value_min=0.,
                                  clip_value_max=max_shape[0])

            x2 = tf.clip_by_value(x2,
                                  clip_value_min=0.,
                                  clip_value_max=max_shape[1])
        return tf.concat(
            [y1[..., None], x1[..., None], y2[..., None], x2[..., None]],
            axis=-1)

    def distribution(self, x):
        x = tf.nn.softmax(tf.reshape(x, [-1, self.reg_max + 1]), axis=-1)
        ln = tf.range(self.reg_max + 1, dtype=tf.float32)
        ln = ln[:, tf.newaxis]
        x = tf.linalg.matmul(x, ln)
        x = tf.reshape(x, [-1, 4])
        return x

    def get_bboxes(self,
                   cls_score,
                   bbox_pred,
                   stride,
                   origin_shape,
                   resize_ratio,
                   rescale=False):

        featmap_size = tf.shape(cls_score)[:2]
        y, x = self.get_single_level_center_point(featmap_size, stride)

        h, w, c = [tf.shape(cls_score)[i] for i in range(3)]
        cls_score = tf.reshape(cls_score, [-1, c])
        scores = tf.math.sigmoid(cls_score)
        ind = tf.where(scores > self.box_score)
        print('-' * 100)
        print(ind)
        # distribution_project
        bbox_pred = self.distribution(bbox_pred) * stride
        center_points = tf.concat([y[:, None], x[:, None]], axis=-1)
        # center_points = tf.tile(center_points[None, :, :], [b, 1, 1])
        nms_pre = 1000
        if h * w > nms_pre:
            max_scores = tf.math.reduce_max(scores, axis=-1)
            _, topk_inds = tf.nn.top_k(max_scores, nms_pre)
            # b_idx = tf.range(b, dtype=tf.int32)
            # b_idx = tf.tile(b_idx[:, None, None], [1, nms_pre, 1])
            # topk_inds = topk_inds[:, :, None]
            # topk_inds = tf.concat([b_idx, topk_inds], axis=-1)
            topk_inds = topk_inds[:, None]
            center_points = tf.gather_nd(center_points, topk_inds)
            bbox_pred = tf.gather_nd(bbox_pred, topk_inds)
            scores = tf.gather_nd(scores, topk_inds)

        # tl = bbox_pred[:, :2]
        # tl = tl[:, ::-1]
        # br = bbox_pred[:, 2:]
        # br = br[:, ::-1]
        # bbox_pred = tf.concat([tl, br], axis=-1)
        bboxes = self.distance2bbox(center_points[:, ::-1],
                                    bbox_pred,
                                    max_shape=self.model_inp_shape)
        return bboxes, scores