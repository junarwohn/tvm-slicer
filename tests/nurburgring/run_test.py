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
current_file_path = os.path.dirname(os.path.realpath(__file__)) + "/"
target = 'cuda'
dev = tvm.cuda()
model_path = current_file_path + "./src/model/test.so"
lib = tvm.runtime.load_module(model_path)
# model = graph_executor.GraphModule(lib['default'](dev))
with open(current_file_path + "./src/graph/test.json", "r") as json_graph_full:
    json_graph_full = json.load(json_graph_full)

with open(current_file_path + "./src/graph/test2.json", "r") as json_graph_full2:
    json_graph_full2 = json.load(json_graph_full2)

model = graph_executor.create(json.dumps(json_graph_full), lib, dev)

print("get outputs", model.get_num_outputs())


model2 = graph_executor.create(json.dumps(json_graph_full2), lib, dev)

print("get outputs", model2.get_num_outputs())
