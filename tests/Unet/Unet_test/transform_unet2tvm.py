import tensorflow as tf
import tvm
import tvm.relay as relay
from tvm.contrib import graph_executor 
import numpy as np
import os
import json
from argparse import ArgumentParser
from tvm.relay.build_module import bind_params_by_name
from tvm.relay.dataflow_pattern import *

np.random.seed(0)
img_size = 256
model_config = [0,0,2,0]
input_data = np.random.normal(0,1,(1,img_size,img_size,3)).astype(np.float32)
input_data = input_data.transpose([0, 3, 1, 2])
model_keras = tf.keras.models.load_model("unet_as_{}_{}_{}_{}.h5".format(*model_config))
shape_dict = {"input_1": input_data.shape}
mod, params = relay.frontend.from_keras(model_keras, shape_dict)
target = 'cuda'
dev = tvm.cuda()
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target, params=params)

lib.export_library("unet_as_{}_{}_{}_{}.so".format(*model_config))
        # param_bytes = tvm.runtime.save_param_dict(lib.get_params())