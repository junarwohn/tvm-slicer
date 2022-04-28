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


path = './graph/unet_cuda_front_512_3_42.json'

with open(path, "r") as json_file:
    json_graph = json.load(json_file)
    show_graph(json_graph, "front")

path = './graph/unet_cuda_back_512_3_42.json'

with open(path, "r") as json_file:
    json_graph = json.load(json_file)
    show_graph(json_graph, "back")

path = './graph/unet_cuda_full_512_3_42.json'

with open(path, "r") as json_file:
    json_graph = json.load(json_file)
    show_graph(json_graph, "full")
