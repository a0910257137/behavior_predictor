import tensorflow as tf
import numpy as np
import cv2
import os
import time
from .core.post_model import PostModel
from pprint import pprint


class BehaviorPredictor:
    def __init__(self, config=None):
        self.config = config
        os.environ['CUDA_VISIBLE_DEVICES'] = self.config['visible_gpu']

        if self.config is not None:
            self.model_dir = self.config['pb_path']
            self.strides = tf.constant(self.config['strides'],
                                       dtype=tf.float32)
            self.scale_factor = self.config['scale_factor']
            self.reg_max = self.config['reg_max']
            self.top_k_n = self.config['top_k_n']

            self.resize = tf.constant(self.config['img_input_size'],
                                      dtype=tf.float32)
            self.iou_thres = self.config['iou_thres']
            self.model = tf.keras.models.load_model(self.model_dir)
            self.post_model = PostModel(self.model, self.strides,
                                        self.scale_factor, self.reg_max,
                                        self.top_k_n, self.iou_thres)
            # imgs = tf.constant(0., shape=(1, 224, 224, 3))
            # preds = self.model(imgs, training=False)

    def pred(self, imgs, origin_shapes):
        imgs = list(
            map(
                lambda x: cv2.resize(x, tuple(self.resize))[:, :, ::-1] /
                255.0, imgs))
        imgs = np.asarray(imgs)
        origin_shapes = np.asarray(origin_shapes)

        imgs = tf.cast(imgs, tf.float32)
        origin_shapes = tf.cast(origin_shapes, tf.float32)
        rets = self.post_model([imgs, origin_shapes])

        # feat_bbox_0 = np.load("../nanodet/feat_bbox_0.npy")
        # feat_bbox_1 = np.load("../nanodet/feat_bbox_1.npy")
        # feat_bbox_2 = np.load("../nanodet/feat_bbox_2.npy")
        # feat_cls_1 = np.load("../nanodet/feat_cls_1.npy")
        # feat_cls_0 = np.load("../nanodet/feat_cls_0.npy")
        # feat_cls_2 = np.load("../nanodet/feat_cls_2.npy")
        # bbox_preds = [
        #     tf.convert_to_tensor(feat_bbox_0),
        #     tf.convert_to_tensor(feat_bbox_1),
        #     tf.convert_to_tensor(feat_bbox_2)
        # ]
        # cls_scores = [
        #     tf.convert_to_tensor(feat_cls_0),
        #     tf.convert_to_tensor(feat_cls_1),
        #     tf.convert_to_tensor(feat_cls_2)
        # ]
        # origin_shapes = tf.convert_to_tensor(origin_shapes)

        # rets = self.post_model(img, training=False)
        return rets

    def batched_nms(self, boxes, scores, idxs):
        class_agnostic = False
        if class_agnostic:
            boxes_for_nms = boxes
        else:
            max_coordinate = tf.math.reduce_max(boxes)
            offsets = tf.cast(idxs, tf.float32) * tf.cast(
                (max_coordinate + 1), tf.float32)
            boxes_for_nms = boxes + offsets[:, None]

        split_thr = 10000
        if len(boxes_for_nms) < split_thr:
            keep = nms(boxes_for_nms, scores, **nms_cfg_)
            boxes = boxes[keep]
            scores = scores[keep]
        else:
            total_mask = scores.new_zeros(scores.size(), dtype=torch.bool)
            for id in torch.unique(idxs):
                mask = (idxs == id).nonzero(as_tuple=False).view(-1)
                keep = nms(boxes_for_nms[mask], scores[mask], **nms_cfg_)
                total_mask[mask[keep]] = True

            keep = total_mask.nonzero(as_tuple=False).view(-1)
            keep = keep[scores[keep].argsort(descending=True)]
            boxes = boxes[keep]
            scores = scores[keep]

        return torch.cat([boxes, scores[:, None]], -1), keep