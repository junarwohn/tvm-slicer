import numpy as np
from tvm.relay import testing
import tvm
from tvm import te
from tvm.contrib import graph_executor
import tvm.testing
import tensorflow as tf
import numpy as np
import tvm.relay as relay
from tvm import relay
from tvm.relay.build_module import bind_params_by_name
from tvm.relay.dataflow_pattern import *
import os
import json
import numpy as np
import pygraphviz as pgv
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--partition_points', '-p', nargs='+', type=int, default=[], help='set partition points')
parser.add_argument('--img_size', '-i', type=int, default=512, help='set image size')
parser.add_argument('--model', '-m', type=str, default='unet', help='name of model')
parser.add_argument('--target', '-t', type=str, default='llvm', help='name of taget')
parser.add_argument('--opt_level', '-o', type=int, default=2, help='set opt_level')
parser.add_argument('--build', '-b', type=int, default=0, help='build model')
parser.add_argument('--add_quantize_layer', '-q', type=int, default=0, help='add int8 quantize layer at sliced edge')
parser.add_argument('--show_size', '-s', type=int, default=0, help='show size')
args = parser.parse_args()

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
                print(src[0], json_data['nodes'][src[0]]['name'], src_size)

                A.add_edge(json_data['nodes'][src[0]]['name'] + '[{}]'.format(src[0]) + '{}'.format(json_data['attrs']['dltype'][1][src[0]]) + "[{}]".format(src_size), node['name'] + '[{}]'.format(node_idx) + '{}'.format(json_data['attrs']['dltype'][1][node_idx]) + "[{}]".format(dst_size))
            else:
                A.add_edge(json_data['nodes'][src[0]]['name'] + '[{}]'.format(src[0]) + '{}'.format(json_data['attrs']['dltype'][1][src[0]]), node['name'] + '[{}]'.format(node_idx) + '{}'.format(json_data['attrs']['dltype'][1][node_idx]))
    if file_name:
        A.draw(file_name + '.png', format='png', prog='dot')
model_config = [2,0,0,0]
path = "./unet_as_{}_{}_{}_{}.json".format(*model_config)
path = "./unet_as_{}_{}_{}_{}_full.json".format(*model_config)
img_path = "./unet_as_{}_{}_{}_{}".format(*model_config)
img_path = "./unet_as_{}_{}_{}_{}".format(*model_config)

with open(path, "r") as json_file:
    json_graph = json.load(json_file)
    show_graph(json_graph, img_path)
