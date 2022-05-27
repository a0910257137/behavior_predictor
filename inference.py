import tensorflow as tf
import numpy as np
import cv2
import os
import time
from .core import *


class BehaviorPredictor:
    def __init__(self, config=None):
        self.config = config
        os.environ['CUDA_VISIBLE_DEVICES'] = self.config['visible_gpu']
        self.gpu_setting(self.config["gpu_fraction"])
        self.model_dir = self.config['pb_path']
        self.top_k_n = self.config['top_k_n']
        self.img_input_size = self.config['img_input_size']
        self.nms_iou_thres = self.config['nms_iou_thres']
        self.resize_shape = np.asarray(config['resize_size'])
        self._model = tf.keras.models.load_model(self.model_dir)
        self.kp_thres = self.config['kp_thres']
        self.n_objs = self.config['n_objs']
        self.mode = self.config['mode']
        if self.mode == 'tflite':
            self.weight_root = self.config["weight_root"]
            interpreter = tf.lite.Interpreter(
                model_path=os.path.join(self.model_dir, "FP32.tflite"))
            self._post_model = Optimize(interpreter, self.weight_root,
                                        self.n_objs, self.top_k_n,
                                        self.kp_thres, self.nms_iou_thres,
                                        self.resize_shape)

        if self.mode == 'centernet':
            self._post_model = CPostModel(self._model, self.n_objs,
                                          self.top_k_n, self.kp_thres,
                                          self.nms_iou_thres,
                                          self.resize_shape)

        elif self.mode == 'offset':
            self._post_model = OffsetPostModel(self._model, self.n_objs,
                                               self.top_k_n, self.kp_thres,
                                               self.nms_iou_thres,
                                               self.resize_shape)
        elif self.mode == '1d_G':
            self._post_model = GPostModel(self._model, self.n_objs,
                                          self.top_k_n, self.kp_thres,
                                          self.nms_iou_thres,
                                          self.resize_shape)
        elif self.mode == 'classification':
            self._post_model = CLSPostModel(self._model)

    def pred(self, imgs, origin_shapes):
        origin_shapes = tf.cast(np.asarray(origin_shapes), tf.float32)
        imgs = list(
            map(
                lambda x: cv2.resize(x,
                                     tuple(self.img_input_size),
                                     interpolation=cv2.INTER_AREA)[..., ::-1] /
                255.0, imgs))
        imgs = np.asarray(imgs)
        imgs = tf.cast(imgs, tf.float32)
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
