from faulthandler import disable
from unittest import result
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
        return post

class UnetCallback(DFPatternCallback):
    # A callback class to rewrite the matched pattern to a batch_norm op.
    def __init__(self, match_node, require_type=False):
        super().__init__(require_type)
        super().__init__(rewrite_once=True)

        self.tuple_get_item_node = is_tuple_get_item(wildcard(), 0)
        self.pattern_1 = self.tuple_get_item_node

        self.pattern = self.pattern_1 
        self.match_node = match_node
        self.counter = 0
        self.tmp = []

    def quant(self, node):
        cast_to_int8 = relay.cast(
            relay.clip(
                relay.round(
                    relay.multiply(node, relay.const(8.0))
                ), 
                a_min=-127.0, a_max=127.0
            ),
            dtype="int8"
        )
        result_node = relay.annotation.stop_fusion(cast_to_int8)
        self.tmp.append(result_node)
        return result_node

    def dequant(self, node):
        cast_to_float32 = relay.divide(
            relay.cast(node, dtype='float32'), relay.const(8.0)
        )
        return cast_to_float32

    def callback(self, pre, post, node_map):
        if self.pattern_1.match(pre):
            if pre in self.match_node:
                print("pat 1")
                return self.dequant(self.quant(post))
        return post


class UnetCallback2(DFPatternCallback):
    # A callback class to rewrite the matched pattern to a batch_norm op.
    def __init__(self, match_node, require_type=False):
        super().__init__(require_type)
        super().__init__(rewrite_once=True)

        # self.tuple_get_item_node = is_tuple_get_item(wildcard(), 0)
        # self.pattern_1 = self.tuple_get_item_node
        self.var2 = wildcard()
        tuple_node = is_tuple([wildcard(), self.var2])
        concat_node = is_op('concatenate')(tuple_node)
        self.pattern = concat_node
        # self.pattern = self.pattern_1 
        self.match_node = match_node
        self.counter = 0
        self.tmp = []

    def quant(self, node):
        cast_to_int8 = relay.cast(
            relay.clip(
                relay.round(
                    relay.multiply(node, relay.const(8.0))
                ), 
                a_min=-127.0, a_max=127.0
            ),
            dtype="int8"
        )
        result_node = relay.annotation.stop_fusion(cast_to_int8)
        self.tmp.append(result_node)
        return result_node

    def dequant(self, node):
        cast_to_float32 = relay.divide(
            relay.cast(node, dtype='float32'), relay.const(8.0)
        )
        return cast_to_float32

    def callback(self, pre, post, node_map):
        if self.pattern.match(pre):
            if pre in self.match_node:
                print("pat 1")
                return self.dequant(self.quant(post))
        return post



parser = ArgumentParser()
parser.add_argument('--partition_points', '-p', nargs='+', type=int, default=[], help='set partition points')
parser.add_argument('--img_size', '-i', type=int, default=512, help='set image size')
parser.add_argument('--model', '-m', type=str, default='unet', help='name of model')
parser.add_argument('--target', '-t', type=str, default='llvm', help='name of taget')
parser.add_argument('--opt_level', '-o', type=int, default=2, help='set opt_level')
parser.add_argument('--build', '-b', type=int, default=0, help='build model')
parser.add_argument('--add_quantize_layer', '-q', type=int, default=0, help='add int8 quantize layer at sliced edge')
args = parser.parse_args()

current_file_path = os.path.dirname(os.path.realpath(__file__)) + "/"

np.random.seed(0)
img_size = args.img_size
input_data = np.random.normal(0,1,(1,img_size,img_size,3)).astype(np.float32)
model_keras = tf.keras.models.load_model(current_file_path + '../../tvm-slicer/src/model/{}_{}.h5'.format(args.model, img_size))

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

# # Inserting quantized layer
# upc = UnetPreProcessCallback()
# rewrite(upc, mod['main'])
# uc = UnetCallback(upc.match_node)
# out = rewrite(uc, mod['main'])


upc = UnetPreProcessCallback()
rewrite(upc, mod['main'])
uc = UnetCallback(upc.match_node)
out = rewrite(uc, mod['main'])

upc = UnetPreProcessCallback()
rewrite(upc, out)
uc2 = UnetCallback2(upc.match_node2)
out = rewrite(uc2, out)

out = relay.Function(out.params, relay.Tuple(uc.tmp + [out.body]), out.ret_type, out.type_params, out.attrs)

with tvm.transform.PassContext(opt_level=args.opt_level):
    lib = relay.build(out, target, params=params)

graph_json_raw = lib['get_graph_json']()

tvm_slicer = TVMSlicer(graph_json_raw)

# graph_json_front_info = tvm_slicer.slice_graph(args.start_point, args.partition_point, is_quantize_sliced=True)
# graph_json_back_info = tvm_slicer.slice_graph(args.partition_point + 1, args.end_point, is_quantize_sliced=True)

# graph_json_front, input_front, output_front = graph_json_front_info
# graph_json_back, input_back, output_back = graph_json_back_info

# Build lib and params
if args.build == 1:
    lib.export_library(current_file_path + "./src/model/{}_{}_full_{}_{}.so".format(args.model, args.target, img_size, args.opt_level))
    param_bytes = tvm.runtime.save_param_dict(lib.get_params())
    with open(current_file_path + "./src/model/{}_{}_full_{}_{}.params".format(args.model, args.target, img_size, args.opt_level), "wb") as f:
        f.write(param_bytes)

# TODO adding final_shape 
# do 'extra' job to 
with open(current_file_path + "./src/graph/{}_{}_full_{}_{}.json".format(args.model, args.target, img_size, args.opt_level), "w") as json_file:
    json_file.write(graph_json_raw)

# json format would be {model}_{target}_{img_size}_{opt_level}_{partition_start}-{partition_end}.json
partition_points = args.partition_points
for i in range(len(partition_points) - 1):
    start_point = partition_points[i]
    end_point = partition_points[i + 1]
    graph_json, input_indexs, output_indexs = tvm_slicer.slice_graph(start_point + 1, end_point, is_quantize_sliced=True)
    with open(current_file_path + "./src/graph/{}_{}_{}_{}_{}-{}.json".format(args.model, args.target, img_size, args.opt_level, start_point, end_point), "w") as json_file:
        graph_json['extra'] = {}
        graph_json['extra']['inputs'] = input_indexs
        graph_json['extra']['outputs'] = output_indexs
        json_file.write(json.dumps(graph_json))


# with open(current_file_path + "./src/graph/{}_{}_front_{}_{}_{}.json".format(args.model, args.target, img_size, args.opt_level, args.partition_point), "w") as json_file:
#     graph_json_front['extra'] = {}
#     graph_json_front['extra']['inputs'] = input_front
#     graph_json_front['extra']['outputs'] = output_front
#     json_file.write(json.dumps(graph_json_front))

# with open(current_file_path + "./src/graph/{}_{}_back_{}_{}_{}.json".format(args.model, args.target, img_size, args.opt_level, args.partition_point), "w") as json_file:
#     graph_json_back['extra'] = {}
#     graph_json_back['extra']['inputs'] = output_front
#     graph_json_back['extra']['outputs'] = output_back
#     json_file.write(json.dumps(graph_json_back))
