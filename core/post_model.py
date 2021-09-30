import tensorflow as tf


class PostModel(tf.keras.Model):
    def __init__(self, model, strides, scale_factor, reg_max, top_k_n,
                 iou_thres, *args, **kwargs):
        super(PostModel, self).__init__(*args, **kwargs)
        self.strides = tf.constant(strides, dtype=tf.float32)
        self.scale_factor = scale_factor
        self.reg_max = reg_max
        self.top_k_n = top_k_n
        self.iou_thres = iou_thres
        self.model = model
        # self.resize = tf.constant(self.config['img_input_size'],
        #                           dtype=tf.float32)

    def call(self, inputs, training=None):
        x, oring_shapes = inputs
        preds = self.model(x, training)
        print(preds)
        cls_scores, bbox_preds = [], []
        for k in preds:
            cls_score, bbox_pred = preds[k]['cls_score'], preds[k]['bbox_pred']
        rets = self.get_bboxes(cls_scores, bbox_preds, origin_shapes)

        return

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
        y1 = points[:, 0] - distance[:, 0]
        x1 = points[:, 1] - distance[:, 1]
        y2 = points[:, 0] + distance[:, 2]
        x2 = points[:, 1] + distance[:, 3]
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
        return tf.concat([y1[:, None], x1[:, None], y2[:, None], x2[:, None]],
                         axis=-1)

    def distribution(self, x):
        x = tf.nn.softmax(tf.reshape(x, [-1, self.reg_max + 1]), axis=-1)
        ln = tf.range(self.reg_max + 1, dtype=tf.float32)
        x = tf.linalg.matmul(x, ln[:, None])
        x = tf.reshape(x, [-1, 4])
        return x

    def get_bboxes(self, cls_scores, bbox_preds, origin_shapes, rescale=False):

        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)
        mlvl_bboxes = []
        mlvl_scores = []
        for stride, cls_score, bbox_pred in zip(self.strides, cls_scores,
                                                bbox_preds):
            cls_score = tf.transpose(cls_score, [0, 2, 3, 1])
            bbox_pred = tf.transpose(bbox_pred, [0, 2, 3, 1])
            cls_score = cls_score[0]
            bbox_pred = bbox_pred[0]
            featmap_size = tf.shape(cls_score)[0:2]

            y, x = self.get_single_level_center_point(featmap_size, stride)
            center_points = tf.concat([y[:, None], x[:, None]], axis=-1)
            scores = tf.nn.sigmoid(cls_score)
            scores = tf.reshape(scores, [-1, 80])
            # distribution_project

            bbox_pred = self.distribution(bbox_pred) * stride
            tl_pred = bbox_pred[:, :2]
            tl_pred = tl_pred[:, ::-1]
            br_pred = bbox_pred[:, 2:]
            br_pred = br_pred[:, ::-1]
            bbox_pred = tf.concat([tl_pred, br_pred], axis=-1)

            nms_pre = 1000
            if tf.shape(scores)[0] > nms_pre:
                max_scores = tf.math.reduce_max(scores, axis=-1)
                _, topk_inds = tf.nn.top_k(max_scores, nms_pre)

                center_points = tf.gather_nd(center_points, topk_inds[:, None])
                bbox_pred = tf.gather_nd(bbox_pred, topk_inds[:, None])
                scores = tf.gather_nd(scores, topk_inds[:, None])

            # center_points,
            # bbox_pred,
            # max_shape=img_shape
            img_shape = tf.cast([256, 320], tf.float32)

            bboxes = self.distance2bbox(center_points,
                                        bbox_pred,
                                        max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)

        mlvl_bboxes = tf.concat(mlvl_bboxes, axis=0)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = tf.concat(mlvl_scores, axis=0)

        # add a dummy background class at the end of all labels

        padding = tf.zeros_like(mlvl_scores[:, :1])
        mlvl_scores = tf.concat([mlvl_scores, padding], axis=-1)
        mlvl_bboxes = mlvl_bboxes[None, :, None, :]
        # mlvl_bboxes = tf.tile(mlvl_bboxes, [1, 80, 1])
        mlvl_scores = mlvl_scores[None, :, :-1]

        nms_reuslt = tf.image.combined_non_max_suppression(mlvl_bboxes,
                                                           mlvl_scores,
                                                           self.n_objs,
                                                           self.n_objs,
                                                           iou_threshold=0.6,
                                                           clip_boxes=False)
        b_bboxes = nms_reuslt[0] * 2.
        mask = nms_reuslt[1] > self.kp_thres

        b_bboxes = tf.concat([
            nms_reuslt[0], nms_reuslt[1][..., None], nms_reuslt[2][..., None]
        ],
                             axis=-1)
        b_bboxes = tf.reshape(b_bboxes[mask], [-1, 2, 6])
        return b_bboxes