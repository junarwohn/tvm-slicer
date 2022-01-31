from ast import arg
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

from tvm.relay.op.tensor import atan

parser = ArgumentParser()
parser.add_argument('--start_point', '-s', type=int, default=0)
parser.add_argument('--end_point', '-e', type=int, default=-1)
parser.add_argument('--partition_point', '-p', type=int, default=0, help='set partition point')
parser.add_argument('--img_size', '-i', type=int, default=512, help='set image size')
parser.add_argument('--model', '-m', type=str, default='unet', help='name of model')
parser.add_argument('--target', '-t', type=str, default='llvm', help='name of taget')
parser.add_argument('--opt_level', '-o', type=int, default=2, help='set opt_level')
parser.add_argument('--graph_mode', '-g', type=int, default=0, help='graph out mode')
parser.add_argument('--build_mode', '-b', type=int, default=0, help='build mode')
parser.add_argument('--model_build', type=int, default=1, help='build model only')
parser.add_argument('--slice_build', type=int, default=1, help='slice model only')
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

if args.graph_mode == 1:
    with tvm.transform.PassContext(opt_level=args.opt_level):
        lib = relay.build(mod, target, params=params)
    graph_json = lib['get_graph_json']()
    with open("./graph/{}_{}_{}_{}_test.json".format(args.model, args.target, img_size, args.opt_level), "w") as json_file:
        # graph_json['extra'] = {}
        # graph_json['extra']['inputs'] = [0,0,0,0]
        # graph_json['extra']['outputs'] = output_front
        json_file.write(json.dumps(json.loads(graph_json)))

else:
    with open("./graph/{}_{}_{}_{}_test.json".format(args.model, args.target, img_size, args.opt_level), "r") as json_graph:
        json_graph = json.load(json_graph)
        with tvm.transform.PassContext(opt_level=args.opt_level):
            lib = relay.build_graph(mod, target=target, target_host=None, params=params, mod_name="default", graph_config=json.dumps(json_graph))
        lib.export_library("./model/{}_{}_{}_{}_test.so".format(args.model, args.target, img_size, args.opt_level))


# graph_json_raw = lib['get_graph_json']()

# if args.model_build == 1:
#     with tvm.transform.PassContext(opt_level=args.opt_level):
#         lib = relay.build(mod, target, params=params)
#         lib.export_library("./model/{}_{}_{}_{}.so".format(args.model, args.target, img_size, args.opt_level))

