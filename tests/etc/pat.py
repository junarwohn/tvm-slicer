import numpy as np

import tvm
from tvm import relay
from tvm.relay.build_module import bind_params_by_name
from tvm.relay.dataflow_pattern import *
from tvm.relay.testing import run_opt_pass

is_conv2d = is_op("nn.conv2d")(wildcard(), wildcard())
# is_unary_elemwise = (wildcard().has_attr({"TOpPattern": K_ELEMWISE}))(wildcard())
# reduction = is_op("add")(wildcard(), wildcard())
reduction = is_op("nn.relu")
diamond = dominates(is_conv2d, wildcard(), reduction)

# Expr
inp = relay.var("input")
weight = relay.var("weight")
conv2d = relay.op.nn.conv2d(inp, weight)
relu = relay.op.nn.relu(conv2d)
relu = relay.op.nn.relu(relu)
leaky_relu = relay.op.nn.leaky_relu(conv2d, alpha=0)
out = relu + leaky_relu

print('-------------------------------')
print("original graph structure")
print(out)
print('-------------------------------')

print("partition [conv2d] ---- ... ---- [leaky_relu]")
print("number of check pattern matched", diamond.match(out))
print("print partitioned graphs")
print(diamond.partition(out))
    # print(g)