from email.policy import default
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
parser.add_argument('--start_point', '-s', type=int, default=0)
parser.add_argument('--end_point', '-e', type=int, default=-1)
parser.add_argument('--partition_point', '-p', type=int, default=0, help='set partition point')
parser.add_argument('--img_size', '-i', type=int, default=256, help='set image size')
parser.add_argument('--model', '-m', type=str, default='unet', help='name of model')
parser.add_argument('--target', '-t', type=str, default='cuda', help='name of taget')
parser.add_argument('--opt_level', '-o', type=int, default=3, help='set opt_level')
parser.add_argument('--whole_build', '-w', type=int, default=0, help='whole model only')
parser.add_argument('--front_build', '-f', type=int, default=0, help='front model only')
parser.add_argument('--back_build', '-b', type=int, default=0, help='back model only')
parser.add_argument('--model_config', '-c', nargs='+', type=int, default=[], help='model config')
args = parser.parse_args()

np.random.seed(0)
img_size = 256
model_config = [2,0,0,0]
input_data = np.random.normal(0,1,(1,img_size,img_size,3)).astype(np.float32)
model_keras = tf.keras.models.load_model("unet_as_{}_{}_{}_{}.h5".format(*model_config))

## tvm result
input_data = input_data.transpose([0, 3, 1, 2])
shape_dict = {"input_1": input_data.shape}
mod, params = relay.frontend.from_keras(model_keras, shape_dict)

target = 'cuda'
dev = tvm.cuda()

upc = UnetPreProcessCallback()
rewrite(upc, mod['main'])
uc = UnetCallback(upc.match_node)
out = rewrite(uc, mod['main'])

upc = UnetPreProcessCallback()
rewrite(upc, out)
uc2 = UnetCallback2(upc.match_node2)
out = rewrite(uc2, out)

out = relay.Function(out.params, relay.Tuple(uc.tmp + [out.body]), out.ret_type, out.type_params, out.attrs)
# out = mod

print(out)

with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(out, target, params=params)

if args.whole_build == 1:
    if True:
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(out, target, params=params)
        lib.export_library("unet_as_{}_{}_{}_{}.so".format(*model_config))
        param_bytes = tvm.runtime.save_param_dict(lib.get_params())
        with open("unet_as_{}_{}_{}_{}.params".format(*model_config), "wb") as f:
            f.write(param_bytes)

graph_json_raw = lib['get_graph_json']()
with open("unet_as_{}_{}_{}_{}.json".format(*model_config), "w") as json_graph:
    json_graph.write(graph_json_raw)
