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

import json
import numpy as np
import pygraphviz as pgv
from argparse import ArgumentParser

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

def show_graph(json_data, file_name=None):
    if type(json_data) == str:
        json_data = json.loads(json_data)
    A = pgv.AGraph(directed=True)
    for node_idx, node in enumerate(json_data['nodes']):
        for src in node['inputs']:
            A.add_edge(json_data['nodes'][src[0]]['name'] + '[{}]'.format(src[0]) + '{}'.format(json_data['attrs']['dltype'][1][src[0]]), node['name'] + '[{}]'.format(node_idx) + '{}'.format(json_data['attrs']['dltype'][1][node_idx]))
            #A.add_edge(json_data['nodes'][src[0]]['name'] + '[{}]'.format(src[0]) + '{}'.format(shape_size(json_data['attrs']['shape'][1][src[0]])) + '{}'.format(json_data['attrs']['dltype'][1][src[0]]), node['name'] + '[{}]'.format(node_idx) + '{}'.format(shape_size(json_data['attrs']['shape'][1][node_idx])) + '{}'.format(json_data['attrs']['dltype'][1][src[0]]))
    if file_name:
        A.draw(file_name + '.png', format='png', prog='dot')


path = './graph/{}_{}_{}_{}_{}_{}.json'
img_path = './img/{}_{}_{}_{}_{}_{}'
# path = './graph/unet_cuda_front_512_3_42.json'

if args.front_build:
    with open(path.format(args.model, args.target, 'front', args.img_size, args.opt_level, args.partition_point), "r") as json_file:
        json_graph = json.load(json_file)
        show_graph(json_graph, img_path.format(args.model, args.target, 'front', args.img_size, args.opt_level, args.partition_point))

if args.back_build:
    with open(path.format(args.model, args.target, 'back', args.img_size, args.opt_level, args.partition_point), "r") as json_file:
        json_graph = json.load(json_file)
        show_graph(json_graph, img_path.format(args.model, args.target, 'back', args.img_size, args.opt_level, args.partition_point))

if args.back_build:
    with open(path.format(args.model, args.target, 'full', args.img_size, args.opt_level, args.partition_point), "r") as json_file:
        json_graph = json.load(json_file)
        show_graph(json_graph, img_path.format(args.model, args.target, 'full', args.img_size, args.opt_level, args.partition_point))
