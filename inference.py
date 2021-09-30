import tensorflow as tf
import numpy as np
import cv2
import os
import time
from .core.post_model import PostModel
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph
from tensorflow.lite.python.util import run_graph_optimizations, get_grappler_config
from pprint import pprint

tf.get_logger().setLevel('ERROR')
tf.__version__


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
            self.resize = self.config['img_input_size']
            self.iou_thres = self.config['iou_thres']
            self.box_score = self.config['box_score']

            self.model = tf.keras.models.load_model(self.model_dir)
            # graph_def = self.frozen_keras_graph(self.model_dir, self.model)
            # pred_func = self.load_pb(
            #     os.path.join(self.model_dir, 'frozen_graph.pb'))
            self.post_model = PostModel(self.resize, self.model, self.strides,
                                        self.scale_factor, self.reg_max,
                                        self.top_k_n, self.iou_thres,
                                        self.box_score)

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
        return rets

    def load_pb(self, model_path):
        def _imports_graph_def():
            tf.compat.v1.import_graph_def(graph_def, name="")

        print(model_path)
        with tf.io.gfile.GFile(model_path, "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            loaded = graph_def.ParseFromString(f.read())

        # Wrap frozen graph to ConcreteFunctions

        wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
        import_graph = wrapped_import.graph
        print("-" * 50)
        print("Frozen model layers: ")
        layers = [op.name for op in import_graph.get_operations()]
        for layer in layers:
            print(layer)
        print("-" * 50)

        return wrapped_import.prune(
            tf.nest.map_structure(import_graph.as_graph_element, ["x:0"]),
            tf.nest.map_structure(import_graph.as_graph_element,
                                  ["Identity:0"]))

    def frozen_keras_graph(self, save_path, model):
        pprint(model.layers[0].inputs)
        inputs = model.layers[0].inputs[0].shape
        dtype = tf.float32
        real_model = tf.function(model).get_concrete_function(
            tf.TensorSpec(inputs, dtype))
        frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(
            real_model)

        input_tensors = [
            tensor for tensor in frozen_func.inputs
            if tensor.dtype != tf.resource
        ]
        output_tensors = frozen_func.outputs
        graph_def = run_graph_optimizations(graph_def,
                                            input_tensors,
                                            output_tensors,
                                            config=get_grappler_config(
                                                ["constfold", "function"]),
                                            graph=frozen_func.graph)
        # frozen_func.graph.as_graph_def()
        os.path.join(save_path, 'frozen_graph')
        tf.io.write_graph(graph_def, save_path, 'frozen_graph.pb')

        return graph_def
