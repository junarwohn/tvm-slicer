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

with tvm.transform.PassContext(opt_level=2):
    lib = relay.build(mod, target, params=params)
    lib.export_library("./model/unet_{}.so".format(img_size))

with open("./graph/{}_{}_front_{}_{}_{}.json".format(args.model, args.target, img_size, args.opt_level, args.partition_point), "r") as json_graph_front:
    json_graph_front = json.load(json_graph_front)
    del json_graph_front['extra']
    with tvm.transform.PassContext(opt_level=args.opt_level):
        lib = relay.build_graph(mod, target=target, target_host=None, params=params, mod_name="default", graph_config=json.dumps(json_graph_front))
    lib.export_library("./model/{}_{}_front_{}_{}_{}.so".format(args.model, args.target, img_size, args.opt_level, args.partition_point))

with open("./graph/{}_{}_back_{}_{}_{}.json".format(args.model, args.target, img_size, args.opt_level, args.partition_point), "r") as json_graph_back:
    json_graph_back = json.load(json_graph_back)
    del json_graph_back['extra']
    with tvm.transform.PassContext(opt_level=args.opt_level):
        lib = relay.build_graph(mod, target=target, target_host=None, params=params, mod_name="default", graph_config=json.dumps(json_graph_back))
    lib.export_library("./model/{}_{}_back_{}_{}_{}.so".format(args.model, args.target, img_size, args.opt_level, args.partition_point))
