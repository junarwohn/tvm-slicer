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
args = parser.parse_args()

np.random.seed(0)
img_size = args.img_size
input_data = np.random.normal(0,1,(1,img_size,img_size,3)).astype(np.float32)
model_keras = tf.keras.models.load_model('./model/{}_{}.h5'.format(args.model, img_size))

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

with tvm.transform.PassContext(opt_level=args.opt_level):
    lib = relay.build(mod, target, params=params)

with relay.quantize.qconfig(calibrate_mode='kl_divergence', weight_scale='max'):
# with relay.quantize.qconfig(calibrate_mode="global_scale", global_scale=8.0):
    mod = relay.quantize.quantize(mod, params)


with tvm.transform.PassContext(opt_level=args.opt_level):
    lib = relay.build(mod, target, params=params)

class JUNOCallback(DFPatternCallback):
    # A callback class to rewrite the matched pattern to a batch_norm op.
    def __init__(self, require_type=False):
        super().__init__(require_type)
        super().__init__(rewrite_once=True)
        self.var1 = wildcard()
        self.var2 = wildcard()
        self.var = relay.var("var")
        self.meta = relay.var("meta")
        self.padding = [1, 1, 1, 1]
        self.channels = 32 
        self.kernel_size = [3, 3]
        self.out_dtype = "int32"
        self.pattern = is_op('nn.max_pool2d')(self.var1)
        self.cnt = 0
        self.target = 0

    def callback(self, pre, post, node_map):
        var1 = node_map[self.var1][0]
        # var2 = node_map[self.var2][0]
        original = relay.nn.max_pool2d(var1, pool_size=(2, 2), strides=(2, 2), padding=(0, 0, 0, 0))

        if self.cnt != self.target:
            print(self.cnt, self.target)
            print("fuck")
            self.cnt += 1
            return original
        else:
            self.cnt += 1
            # return var1 * var2
            # original = relay.nn.bias_add(var1, var2)
            cast_to_int8 = relay.cast(
                relay.clip(
                    relay.round(
                        relay.multiply(original, relay.const(16.0))
                    ), 
                    a_min=-127.0, a_max=127.0
                ),
                dtype="int8"
            )
            print(cast_to_int8)
            cast_to_float32 = relay.cast(
                relay.clip(
                    relay.right_shift(
                        relay.add(relay.cast(relay.annotation.stop_fusion(cast_to_int8), dtype='int32'), relay.const(512)),
                        relay.const(10)),
                    a_min=-127.0, a_max=127.0), 
                dtype="float32"
            )
            return cast_to_float32



print(mod.astext())
graph_json_raw = lib['get_graph_json']()

print(graph_json_raw)


tvm_slicer = TVMSlicer(graph_json_raw)
#parser.add_argument('--start_point', type=int, default=0)
#parser.add_argument('--end_point', type=int, default=-1)
#parser.add_argument('--partition_point', type=int, default=0, help='set partition point')

graph_json_front_info = tvm_slicer.slice_graph(args.start_point, args.partition_point)
graph_json_back_info = tvm_slicer.slice_graph(args.partition_point, args.end_point)

#graph_json_back_info = TVMSlicer(graph_json_raw, [[0,10],[10,121]]).get_graph()
#graph_json_front_info, graph_json_back_info = TVMSlicer(graph_json_raw, [[0,9],[9,111]]).get_graph()

graph_json_front, input_front, output_front = graph_json_front_info
graph_json_back, input_back, output_back = graph_json_back_info

# TODO adding final_shape 
# do 'extra' job to 


with open("./graph/{}_{}_full_{}_{}_{}.json".format(args.model, args.target, img_size, args.opt_level, args.partition_point), "w") as json_file:
    json_file.write(graph_json_raw)

with open("./graph/{}_{}_front_{}_{}_{}.json".format(args.model, args.target, img_size, args.opt_level, args.partition_point), "w") as json_file:
    graph_json_front['extra'] = {}
    graph_json_front['extra']['inputs'] = input_front
    graph_json_front['extra']['outputs'] = output_front
    json_file.write(json.dumps(graph_json_front))


with open("./graph/{}_{}_back_{}_{}_{}.json".format(args.model, args.target, img_size, args.opt_level, args.partition_point), "w") as json_file:
    graph_json_back['extra'] = {}
    graph_json_back['extra']['inputs'] = output_front
    graph_json_back['extra']['outputs'] = output_back
    json_file.write(json.dumps(graph_json_back))
