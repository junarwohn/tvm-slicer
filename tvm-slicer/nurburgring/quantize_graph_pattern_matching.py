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

    def callback(self, pre, post, node_map):
        var2 = node_map[self.var2][0]
        self.match_node.append(var2)
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

parser = ArgumentParser()
parser.add_argument('--start_point', '-s', type=int, default=0)
parser.add_argument('--end_point', '-e', type=int, default=-1)
parser.add_argument('--partition_point', '-p', type=int, default=0, help='set partition point')
parser.add_argument('--img_size', '-i', type=int, default=512, help='set image size')
parser.add_argument('--model', '-m', type=str, default='unet', help='name of model')
parser.add_argument('--target', '-t', type=str, default='llvm', help='name of taget')
parser.add_argument('--opt_level', '-o', type=int, default=2, help='set opt_level')
parser.add_argument('--whole_build', '-w', type=int, default=0, help='whole model only')
parser.add_argument('--front_build', '-f', type=int, default=0, help='front model only')
parser.add_argument('--back_build', '-b', type=int, default=0, help='back model only')
parser.add_argument('--add_quantize_layer', '-q', type=int, default=0, help='add int8 quantize layer at sliced edge')
args = parser.parse_args()

np.random.seed(0)
img_size = args.img_size
input_data = np.random.normal(0,1,(1,img_size,img_size,3)).astype(np.float32)
model_keras = tf.keras.models.load_model('../src/model/{}_{}.h5'.format(args.model, img_size))

## tvm result
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
rewrite(upc, mod['main'])
uc = UnetCallback(upc.match_node)
out = rewrite(uc, mod['main'])

out = relay.Function(out.params, relay.Tuple(uc.tmp + [out.body]), out.ret_type, out.type_params, out.attrs)

with tvm.transform.PassContext(opt_level=args.opt_level):
    lib = relay.build(out, target, params=params)

graph_json_raw = lib['get_graph_json']()

tvm_slicer = TVMSlicer(graph_json_raw)

graph_json_front_info = tvm_slicer.slice_graph(args.start_point, args.partition_point, is_quantize_sliced=True)
graph_json_back_info = tvm_slicer.slice_graph(args.partition_point, args.end_point, is_quantize_sliced=True)

graph_json_front, input_front, output_front = graph_json_front_info
graph_json_back, input_back, output_back = graph_json_back_info

# TODO adding final_shape 
# do 'extra' job to 

with open("./src/graph/{}_{}_full_{}_{}_{}.json".format(args.model, args.target, img_size, args.opt_level, args.partition_point), "w") as json_file:
    json_file.write(graph_json_raw)

with open("./src/graph/{}_{}_front_{}_{}_{}.json".format(args.model, args.target, img_size, args.opt_level, args.partition_point), "w") as json_file:
    graph_json_front['extra'] = {}
    graph_json_front['extra']['inputs'] = input_front
    graph_json_front['extra']['outputs'] = output_front
    json_file.write(json.dumps(graph_json_front))

with open("./src/graph/{}_{}_back_{}_{}_{}.json".format(args.model, args.target, img_size, args.opt_level, args.partition_point), "w") as json_file:
    graph_json_back['extra'] = {}
    graph_json_back['extra']['inputs'] = output_front
    graph_json_back['extra']['outputs'] = output_back
    json_file.write(json.dumps(graph_json_back))
