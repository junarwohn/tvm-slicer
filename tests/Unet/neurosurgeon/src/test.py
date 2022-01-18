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

parser = ArgumentParser()
parser.add_argument('--start_point', type=int, default=0)
parser.add_argument('--end_point', type=int, default=-1)
parser.add_argument('--partition_point', type=int, default=0, help='set partition point')
parser.add_argument('--img_size', type=int, default=512, help='set image size')
parser.add_argument('--model', type=str, default='unet', help='name of model')
parser.add_argument('--target', type=str, default='llvm', help='name of taget')
parser.add_argument('--opt_level', type=int, default=2, help='set opt_level')
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

graph_json_raw = lib['get_graph_json']()
tvm_slicer = TVMSlicer(graph_json_raw)
#graph_json_front_info = tvm_slicer.slice_graph(0, 10)
#graph_json_back_info = tvm_slicer.slice_graph(10, 121)

print(*tvm_slicer.get_all_intermediate_node())

