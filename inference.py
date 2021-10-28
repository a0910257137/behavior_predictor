import tensorflow as tf
import numpy as np
import cv2
import os
import time
from pprint import pprint
from .core.anchor_model import APostModel
from .core.centernet_model import CPostModel


class BehaviorPredictor:
    def __init__(self, config=None):
        self.config = config
        os.environ['CUDA_VISIBLE_DEVICES'] = self.config['visible_gpu']
        self.gpu_setting(self.config["gpu_fraction"])
        if self.config is not None:
            self.mode = self.config['mode']
            self.top_k_n = self.config['top_k_n']
            self.model_dir = self.config['pb_path']
            self.img_input_size = self.config['img_input_size']
            self.nms_iou_thres = self.config['nms_iou_thres']
            self.resize_shape = np.asarray(config['resize_size'])
            self._model = tf.keras.models.load_model(self.model_dir)
            if self.mode == 'anchor':
                self.strides = tf.constant(self.config['strides'],
                                           dtype=tf.float32)
                self.scale_factor = self.config['scale_factor']
                self.reg_max = self.config['reg_max']
                self.nms_iou_thres = self.config['nms_iou_thres']
                self.box_score = self.config['box_score']
                self._post_model = APostModel(self.resize_shape, self._model,
                                              self.strides, self.scale_factor,
                                              self.reg_max, self.top_k_n,
                                              self.nms_iou_thres,
                                              self.box_score)

            elif self.mode == 'centernet':
                self.kp_thres = self.config['kp_thres']
                self.n_objs = self.config['n_objs']
                self.k_pairings = self.config['k_pairings']
                self._post_model = CPostModel(self._model, self.n_objs,
                                              self.k_pairings, self.top_k_n,
                                              self.kp_thres,
                                              self.nms_iou_thres,
                                              self.resize_shape)

    def pred(self, imgs, origin_shapes):
        imgs = list(
            map(
                lambda x: cv2.resize(x,
                                     tuple(self.img_input_size),
                                     interpolation=cv2.INTER_AREA)[:, :, ::-1]
                / 255.0, imgs))
        imgs = np.asarray(imgs)
        origin_shapes = np.asarray(origin_shapes)
        imgs = tf.cast(imgs, tf.float32)
        origin_shapes = tf.cast(origin_shapes, tf.float32)
        star_time = time.time()
        rets = self._post_model([imgs, origin_shapes], training=False)
        # print("%.3f" % (time.time() - star_time))
        return rets

    def gpu_setting(self, fraction):
        gpus = tf.config.experimental.list_physical_devices('GPU')
        gpu_config = tf.compat.v1.ConfigProto()
        gpu_config.gpu_options.allow_growth = True
        gpu_config.gpu_options.per_process_gpu_memory_fraction = fraction
        tf.compat.v1.keras.backend.set_session(
            tf.compat.v1.Session(config=gpu_config))
        for i in range(len(gpus)):
            tf.config.experimental.set_memory_growth(gpus[i], True)
