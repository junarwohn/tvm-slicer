from faulthandler import disable
from unittest import result
# from SlicingMachine import TVMSlicer
from mozer.slicer.SlicingMachine import TVMSlicer
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import tvm
import tvm.relay as relay
from tvm.contrib import graph_executor 
import numpy as np
import json
import pygraphviz as pgv
from argparse import ArgumentParser
from tvm.relay.build_module import bind_params_by_name
from tvm.relay.dataflow_pattern import *
from tvm import rpc
from tvm.autotvm.measure.measure_methods import set_cuda_target_arch
from tensorflow import keras
import pygraphviz as pgv

def show_graph(json_data, file_name=None):
    if type(json_data) == str:
        json_data = json.loads(json_data)
    A = pgv.AGraph(directed=True)
    for node_idx, node in enumerate(json_data['nodes']):
        for src in node['inputs']:
            # if args.show_size == 1:
            if 1 == 1:
                src_size = 1
                for i in json_data['attrs']['shape'][1][src[0]]:
                    src_size = src_size * i
                
                dst_size = 1
                for i in json_data['attrs']['shape'][1][node_idx]:
                    dst_size = dst_size * i
                # print(src[0], json_data['nodes'][src[0]]['name'], src_size)

                A.add_edge(json_data['nodes'][src[0]]['name'] + '[{}]'.format(src[0]) + '{}'.format(json_data['attrs']['dltype'][1][src[0]]) + "[{}]".format(src_size), node['name'] + '[{}]'.format(node_idx) + '{}'.format(json_data['attrs']['dltype'][1][node_idx]) + "[{}]".format(dst_size))
            else:
                A.add_edge(json_data['nodes'][src[0]]['name'] + '[{}]'.format(src[0]) + '{}'.format(json_data['attrs']['dltype'][1][src[0]]), node['name'] + '[{}]'.format(node_idx) + '{}'.format(json_data['attrs']['dltype'][1][node_idx]))
    if file_name:
        A.draw(file_name + '.png', format='png', prog='dot')

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
                # print("pat 1")
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
                # print("pat 1")
                return self.dequant(self.quant(post))
        return post


class UnetMaxPool2dCallback(DFPatternCallback):
    # A callback class to rewrite the matched pattern to a batch_norm op.
    def __init__(self, require_type=False):
        super().__init__(require_type)
        super().__init__(rewrite_once=True)

        max_pool2d_node = is_op('nn.max_pool2d')(wildcard())
        self.pattern = max_pool2d_node
        self.match_node = []

    def callback(self, pre, post, node_map):
        self.match_node.append(pre)
        return post


class UnetCallback3(DFPatternCallback):
    # A callback class to rewrite the matched pattern to a batch_norm op.
    def __init__(self, match_node, require_type=False):
        super().__init__(require_type)
        super().__init__(rewrite_once=True)

        # self.tuple_get_item_node = is_tuple_get_item(wildcard(), 0)
        # self.pattern_1 = self.tuple_get_item_node
        max_pool2d_node = is_op('nn.max_pool2d')(wildcard())
        self.pattern = max_pool2d_node
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
        # print("match pool2d")

        if self.pattern.match(pre):
            if pre in self.match_node:
                # print("pat 1")
                return self.dequant(self.quant(post))
        return post

class UnetLeakyReLUCallback(DFPatternCallback):
    # A callback class to rewrite the matched pattern to a batch_norm op.
    def __init__(self, require_type=False):
        super().__init__(require_type)
        super().__init__(rewrite_once=True)

        leaky_relu_node = is_op('nn.leaky_relu')(wildcard())
        self.pattern = leaky_relu_node
        self.match_node = []

    def callback(self, pre, post, node_map):
        self.match_node.append(pre)
        return post


class UnetCallback4(DFPatternCallback):
    # A callback class to rewrite the matched pattern to a batch_norm op.
    def __init__(self, match_node, require_type=False):
        super().__init__(require_type)
        super().__init__(rewrite_once=True)

        # self.tuple_get_item_node = is_tuple_get_item(wildcard(), 0)
        # self.pattern_1 = self.tuple_get_item_node
        leaky_relu_node = is_op('nn.leaky_relu')(wildcard())
        self.pattern = leaky_relu_node
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
        # print("match leaky_relu_node")

        if self.pattern.match(pre):
            if pre in self.match_node:
                # print("pat 1")
                return self.dequant(self.quant(post))
        return post

class Int8Collector(DFPatternCallback):
    # A callback class to rewrite the matched pattern to a batch_norm op.
    def __init__(self, require_type=False):
        super().__init__(require_type)
        super().__init__(rewrite_once=True)

        int8_cast_node = is_op('cast')(wildcard()).has_attr({'dtype': 'int8'})

        self.pattern = int8_cast_node
        self.match_node = []

    def callback(self, pre, post, node_map):
        # print(pre)
        self.match_node.append(pre)
        return post


parser = ArgumentParser()
parser.add_argument('--partition_points', '-p', nargs='+', type=str, default=[], help='set partition points')
parser.add_argument('--img_size', '-i', type=int, default=256, help='set image size')
parser.add_argument('--model', '-m', type=str, default='unet', help='name of model')
parser.add_argument('--target', '-t', type=str, default='cuda', help='name of taget')
parser.add_argument('--opt_level', '-o', type=int, default=3, help='set opt_level')
parser.add_argument('--build', '-b', type=int, default=0, help='build model')
parser.add_argument('--quantization_level', '-q', type=int, default=0, help='set quantization level')
parser.add_argument('--model_config', '-c', nargs='+', type=int, default=0, help='set partition point')
parser.add_argument('--jetson', '-j', type=int, default=0, help='jetson')
parser.add_argument('--base_path', type=str, default=os.environ['TS_DATA_PATH'] + "/tf_model/unet_v1/best/", help='path setting')
args = parser.parse_args()

# set_cuda_target_arch("sm_62")
# set_cuda_target_arch("sm_72")

# set path
base_path = args.base_path

current_file_path = os.path.dirname(os.path.realpath(__file__)) + "/"
model_config = args.model_config
is_jetson = args.jetson
np.random.seed(0)
img_size = args.img_size
input_data = np.random.normal(0,1,(1,img_size,img_size,3)).astype(np.float32)
model_keras = tf.keras.models.load_model('/'.join([base_path, "UNet_M[{}-{}-{}-{}].h5"]).format(*model_config))

# tvm result
input_data = input_data.transpose([0, 3, 1, 2])
shape_dict = {"input_1": input_data.shape}
print(type(model_keras).__module__)
mod, params = relay.frontend.from_keras(model_keras, shape_dict)



if is_jetson == 1:
    # target = tvm.target.Target("nvidia/jetson-nano")
    target = tvm.target.Target("nvidia/jetson-agx-xavier")
    # target = tvm.target.Target("nvidia/jetson-tx2")
    assert target.kind.name == "cuda"
    # assert target.attrs["arch"] == "sm_62"
    # target.attrs["arch"] = "sm_62"
    assert target.attrs["shared_memory_per_block"] == 49152
    assert target.attrs["max_threads_per_block"] == 1024
    assert target.attrs["thread_warp_size"] == 32
    # assert target.attrs["registers_per_block"] == 65536
else:
    if args.target == 'llvm':
        target = 'llvm'
        dev = tvm.cpu()
    elif args.target == 'cuda':
        target = 'cuda'
        dev = tvm.cuda()
    elif args.target == 'opencl':
        target = 'opencl'
        dev = tvm.opencl()

quantization_level = args.quantization_level

upc = UnetPreProcessCallback()
out = rewrite(upc, mod['main'])

if quantization_level == 0:
    maxpool = UnetMaxPool2dCallback()
    rewrite(maxpool, out)
    leakyrelu = UnetLeakyReLUCallback()
    rewrite(leakyrelu, out)
    callnodes = upc.match_node + upc.match_node2 + maxpool.match_node + leakyrelu.match_node + [out.body]
    callnodes_str = [str(node) for node in callnodes]
    callnodes_str = list(set(callnodes_str))
    callnodes_str.sort(key=lambda x: len(x))
    callnodes_str = callnodes_str[::-1]
    out_nodes = [None for i in range(len(callnodes_str))]
    for node in callnodes:
        out_nodes[callnodes_str.index(str(node))] = node
    # out = relay.Function(out.params, relay.Tuple(upc.match_node + upc.match_node2 + maxpool.match_node + [out.body]), out.ret_type, out.type_params, out.attrs)
    out = relay.Function(out.params, relay.Tuple(out_nodes), out.ret_type, out.type_params, out.attrs)
else:
    uc = UnetCallback(upc.match_node)
    out = rewrite(uc, mod['main'])
    upc = UnetPreProcessCallback()
    rewrite(upc, out)
    uc2 = UnetCallback2(upc.match_node2)
    out = rewrite(uc2, out)
    
    if quantization_level == 1:
        callnodes = uc.tmp + [out.body]
        callnodes_str = [str(node) for node in callnodes]
        callnodes_str = list(set(callnodes_str))
        callnodes_str.sort(key=lambda x: len(x))
        callnodes_str = callnodes_str[::-1]
        out_nodes = [None for i in range(len(callnodes_str))]
        for node in callnodes:
            out_nodes[callnodes_str.index(str(node))] = node
        # out = relay.Function(out.params, relay.Tuple(uc.tmp + [out.body]), out.ret_type, out.type_params, out.attrs)
        out = relay.Function(out.params, relay.Tuple(out_nodes), out.ret_type, out.type_params, out.attrs)

    elif quantization_level == 2:

        upc = UnetMaxPool2dCallback()
        rewrite(upc, out)
        # print(len(upc.match_node))
        uc2 = UnetCallback3(upc.match_node)
        out = rewrite(uc2, out)

        upc = UnetLeakyReLUCallback()
        rewrite(upc, out)
        # print(len(upc.match_node))
        uc2 = UnetCallback4(upc.match_node)
        out = rewrite(uc2, out)

        int8_collector = Int8Collector()
        rewrite(int8_collector, out)
        
        callnodes = int8_collector.match_node + [out.body]
        callnodes_str = [str(node) for node in callnodes]
        callnodes_str = list(set(callnodes_str))
        callnodes_str.sort(key=lambda x: len(x))
        callnodes_str = callnodes_str[::-1]
        out_nodes = [None for i in range(len(callnodes_str))]
        for node in callnodes:
            out_nodes[callnodes_str.index(str(node))] = node
        # out = relay.Function(out.params, relay.Tuple(int8_collector.match_node + [out.body]), out.ret_type, out.type_params, out.attrs)
        # out = relay.Function(out.params, relay.Tuple(out_nodes), out.ret_type, out.type_params, out.attrs)

# print(out)

os.sys.path.append(os.path.join(os.environ['TVM_HOME'] + "/tests/python/relay"))
from test_pipeline_executor import graph_split

###########################################
# Splitting the network into two subgraphs.
split_config = [{"op_name": "annotation.stop_fusion", "op_index": 2}]
subgraphs = graph_split(out, split_config, params)

for g in subgraphs:
    print(g)
    print("-=---------------------------------")

for i, out in enumerate(subgraphs):
    with tvm.transform.PassContext(opt_level=args.opt_level):
        lib = relay.build(out, target, params=params)
        show_graph(lib['get_graph_json'](), f"asd_{i}")
        print(json.loads(lib['get_graph_json']())['heads'])
#     # lib = relay.build(out, target, params=params, target_host="llvm -mtriple=aarch64-linux-gnueabihf -device=arm_cpu")
#     # lib = relay.build(out, target='cuda -arch=sm_72 -model=tx2', params=params, target_host="llvm -mtriple=aarch64-linux-gnueabihf -device=arm_cpu")
#     # lib = relay.build(out, target='cuda -arch=sm_72 -model=tx2', params=params, target_host='llvm -mtriple=aarch64-linux-gnueabihf -device=arm_cpu')

# graph_json_raw = lib['get_graph_json']()

# tvm_slicer = TVMSlicer(graph_json_raw)

# # Build lib and params
# if args.build == 1:
#     if is_jetson == 1:
#         model_format = '/'.join([base_path, "UNet_M[{}-{}-{}-{}]_Q[{}]_full_jetson.so"])
#     else:
#         model_format = '/'.join([base_path, "UNet_M[{}-{}-{}-{}]_Q[{}]_full.so"])
#     lib.export_library(model_format.format(*model_config, quantization_level))
#     # lib.export_library(model_format.format(*model_config, quantization_level), cc=f'/usr/bin/aarch64-linux-gnu-g++')
#     # lib.export_library(lib_path, cc=f'/usr/bin/aarch64-linux-gnu-g++')



#     if is_jetson == 1:
#         param_format = '/'.join([base_path, "UNet_M[{}-{}-{}-{}]_Q[{}]_full_jetson.params"])
#     else:
#         param_format = '/'.join([base_path, "UNet_M[{}-{}-{}-{}]_Q[{}]_full.params"])
#     param_bytes = tvm.runtime.save_param_dict(lib.get_params())
#     with open(param_format.format(*model_config, quantization_level), "wb") as f:
#         f.write(param_bytes)

# # TODO adding final_shape 
# # do 'extra' job to 
# if is_jetson == 1:
#     json_format = '/'.join([base_path, "UNet_M[{}-{}-{}-{}]_Q[{}]_full_jetson.json"])
# else:
#     json_format = '/'.join([base_path, "UNet_M[{}-{}-{}-{}]_Q[{}]_full.json"])

# with open(json_format.format(*model_config, quantization_level), "w") as json_file:
#     json_file.write(graph_json_raw)

# graph_info = json.loads(graph_json_raw)
# # print(len(graph_info['nodes'])-1)

# if is_jetson == 1:
#     json_format = '/'.join([base_path, "UNet_M[{}-{}-{}-{}]_Q[{}]_S[{}-{}]_jetson.json"])
# else:
#     json_format = '/'.join([base_path, "UNet_M[{}-{}-{}-{}]_Q[{}]_S[{}-{}].json"])

# with open(json_format.format(*model_config, quantization_level, 0, len(graph_info['nodes'])-1), "w") as json_file:
#     graph_json, input_indexs, output_indexs = tvm_slicer.slice_graph([1], [len(graph_info['nodes'])-1], is_quantize_sliced=True)
#     graph_json['extra'] = {}
#     graph_json['extra']['inputs'] = input_indexs
#     graph_json['extra']['outputs'] = output_indexs
#     json_file.write(json.dumps(graph_json))
