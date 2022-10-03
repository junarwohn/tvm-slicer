from SlicingMachine import TVMSlicer
import tensorflow as tf
import tvm
import tvm.relay as relay
from tvm.contrib import graph_executor 
import numpy as np
import os
import json
import pygraphviz as pgv
from argparse import ArgumentParser
from tvm.relay.build_module import bind_params_by_name
from tvm.relay.dataflow_pattern import *

class UnetPreProcessCallback(DFPatternCallback):
    # A callback class to rewrite the matched pattern to a batch_norm op.
    def __init__(self, require_type=False):
        super().__init__(require_type)
        super().__init__(rewrite_once=True)

        self.var2 = wildcard()
        tuple_node = is_tuple([wildcard(), self.var2])
        concat_node = is_op('concatenate')(tuple_node)
        self.pattern = concat_node
        self.match_node = []
        self.match_node2 = []

    def callback(self, pre, post, node_map):
        var2 = node_map[self.var2][0]
        self.match_node.append(var2)
        self.match_node2.append(pre)
        return pre

parser = ArgumentParser()
parser.add_argument('--partition_points', '-p', nargs='+', type=int, default=[], help='set partition points')
parser.add_argument('--img_size', '-i', type=int, default=256, help='set image size')
parser.add_argument('--model', '-m', type=str, default='unet', help='name of model')
parser.add_argument('--target', '-t', type=str, default='cuda', help='name of taget')
parser.add_argument('--opt_level', '-o', type=int, default=3, help='set opt_level')
parser.add_argument('--build', '-b', type=int, default=0, help='build model')
parser.add_argument('--add_quantize_layer', '-q', type=int, default=0, help='add int8 quantize layer at sliced edge')
parser.add_argument('--model_config', '-c', nargs='+', type=int, default=0, help='set partition point')
args = parser.parse_args()

model_config = args.model_config

np.random.seed(0)
img_size = args.img_size
input_data = np.random.normal(0,1,(1,img_size,img_size,3)).astype(np.float32)
model_keras = tf.keras.models.load_model("unet_as_{}_{}_{}_{}.h5".format(*model_config))

# tvm result
input_data = input_data.transpose([0, 3, 1, 2])
shape_dict = {"input_1": input_data.shape}
mod, params = relay.frontend.from_keras(model_keras, shape_dict)
if args.target == 'llvm':
    target = 'llvm'
    dev = tvm.cpu()
elif args.target == 'cuda':
    target = 'cuda'
    dev = tvm.cuda()
elif args.target == 'opencl':
    target = 'opencl'
    dev = tvm.opencl()

upc = UnetPreProcessCallback()
out = rewrite(upc, mod['main'])

out = relay.Function(out.params, relay.Tuple(upc.match_node + [out.body]), out.ret_type, out.type_params, out.attrs)

with tvm.transform.PassContext(opt_level=args.opt_level):
    lib = relay.build(out, target, params=params)

graph_json_raw = lib['get_graph_json']()

tvm_slicer = TVMSlicer(graph_json_raw)

# Build lib and params
if args.build == 1:
    lib.export_library("unet_as_{}_{}_{}_{}_full.so".format(*model_config))
    param_bytes = tvm.runtime.save_param_dict(lib.get_params())
    with open("unet_as_{}_{}_{}_{}_full.params".format(*model_config), "wb") as f:
        f.write(param_bytes)

# TODO adding final_shape 
# do 'extra' job to 
with open("unet_as_{}_{}_{}_{}_full.json".format(*model_config), "w") as json_file:
    json_file.write(graph_json_raw)

# json format would be {model}_{target}_{img_size}_{opt_level}_{partition_start}-{partition_end}.json
partition_points = args.partition_points
for i in range(len(partition_points) - 1):
    start_point = partition_points[i]
    end_point = partition_points[i + 1]
    graph_json, input_indexs, output_indexs = tvm_slicer.slice_graph(start_point + 1, end_point, is_quantize_sliced=True)
    with open("unet_as_{}_{}_{}_{}_{}-{}.json".format(*model_config, start_point, end_point), "w") as json_file:
        graph_json['extra'] = {}
        graph_json['extra']['inputs'] = input_indexs
        graph_json['extra']['outputs'] = output_indexs
        json_file.write(json.dumps(graph_json))
