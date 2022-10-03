import os
import sys

import numpy as np

import tvm
from tvm.relay.backend import te_compiler
from tvm.relay.backend.runtime import Runtime
import tvm.relay.testing
import tvm.relay.op as reg
from tvm import relay
from tvm import runtime as tvm_runtime
from tvm.relay import transform
from tvm.relay.testing import byoc
from tvm.contrib import utils
from tvm.relay.expr_functor import ExprMutator
from tvm.relay.op.annotation import compiler_begin, compiler_end
from tvm.relay.op.contrib.register import get_pattern_table
from tvm.relay.build_module import bind_params_by_name

def set_func_attr(func, compile_name, symbol_name):
    func = func.with_attr("Primitive", tvm.tir.IntImm("int32", 1))
    func = func.with_attr("Inline", tvm.tir.IntImm("int32", 1))
    func = func.with_attr("Compiler", compile_name)
    func = func.with_attr("global_symbol", symbol_name)
    return func



def create_graph():
    data = relay.var("data", relay.TensorType((1, 3, 224, 224), "float32"))
    weight = relay.var("weight", relay.TensorType((16, 3, 3, 3), "float32"))
    bn_gamma = relay.var("bn_gamma", relay.TensorType((16,), "float32"))
    bn_beta = relay.var("bn_beta", relay.TensorType((16,), "float32"))
    bn_mean = relay.var("bn_mean", relay.TensorType((16,), "float32"))
    bn_var = relay.var("bn_var", relay.TensorType((16,), "float32"))

    data_cb = compiler_begin(data, "test_target")
    weight_cb = compiler_begin(weight, "test_target")
    bn_gamma_cb = compiler_begin(bn_gamma, "test_target")
    bn_beta_cb = compiler_begin(bn_beta, "test_target")
    bn_mean_cb = compiler_begin(bn_mean, "test_target")
    bn_var_cb = compiler_begin(bn_var, "test_target")

    conv_o = relay.nn.conv2d(
        data=data_cb, weight=weight_cb, kernel_size=(3, 3), channels=16, padding=(1, 1)
    )

    bn_o = relay.nn.batch_norm(conv_o, bn_gamma_cb, bn_beta_cb, bn_mean_cb, bn_var_cb)

    relu_o = relay.nn.relu(bn_o[0])
    relu_o_ce = compiler_end(relu_o, "test_target")

    bn_omean = bn_o[1]
    rebn_omean_ce = compiler_end(bn_omean, "test_target")
    bn_ovar = bn_o[2]
    bn_ovar_ce = compiler_end(bn_ovar, "test_target")

    dummy_mean_abs = relay.abs(rebn_omean_ce)
    dummy_ovar_abs = relay.abs(bn_ovar_ce)
    dummy_tuple = relay.Tuple((relu_o_ce, dummy_mean_abs, dummy_ovar_abs))

    func = relay.Function([data, weight, bn_gamma, bn_beta, bn_mean, bn_var], dummy_tuple)
    return func



def expected():
    mod = tvm.IRModule()

    # function 0
    data = relay.var("test_target_0_i0", relay.TensorType((1, 3, 224, 224), "float32"))
    weight = relay.var("test_target_0_i1", relay.TensorType((16, 3, 3, 3), "float32"))
    bn_gamma = relay.var("test_target_0_i2", relay.TensorType((16,), "float32"))
    bn_beta = relay.var("test_target_0_i3", relay.TensorType((16,), "float32"))
    bn_mean = relay.var("test_target_0_i4", relay.TensorType((16,), "float32"))
    bn_var = relay.var("test_target_0_i5", relay.TensorType((16,), "float32"))

    conv_o = relay.nn.conv2d(
        data=data, weight=weight, kernel_size=(3, 3), channels=16, padding=(1, 1)
    )

    bn_o = relay.nn.batch_norm(conv_o, bn_gamma, bn_beta, bn_mean, bn_var)

    relu_o = relay.nn.relu(bn_o[0])
    tuple_o = relay.Tuple((relu_o, bn_o[1], bn_o[2]))

    func0 = relay.Function([data, weight, bn_gamma, bn_beta, bn_mean, bn_var], tuple_o)
    func0 = set_func_attr(func0, "test_target", "tvmgen_default_test_target_main_0")
    gv0 = relay.GlobalVar("tvmgen_default_test_target_main_0")
    mod[gv0] = func0
    mod = relay.transform.InferType()(mod)

    # body
    data = relay.var("data", relay.TensorType((1, 3, 224, 224), "float32"))
    weight = relay.var("weight", relay.TensorType((16, 3, 3, 3), "float32"))
    bn_gamma = relay.var("bn_gamma", relay.TensorType((16,), "float32"))
    bn_beta = relay.var("bn_beta", relay.TensorType((16,), "float32"))
    bn_mean = relay.var("bn_mean", relay.TensorType((16,), "float32"))
    bn_var = relay.var("bn_var", relay.TensorType((16,), "float32"))

    f0_o = gv0(data, weight, bn_gamma, bn_beta, bn_mean, bn_var)
    f0_relu_o = relay.TupleGetItem(f0_o, 0)
    f0_mean_o = relay.TupleGetItem(f0_o, 1)
    f0_var_o = relay.TupleGetItem(f0_o, 2)

    f0_mean_abs = relay.abs(f0_mean_o)
    f0_var_abs = relay.abs(f0_var_o)
    main_tuple = relay.Tuple((f0_relu_o, f0_mean_abs, f0_var_abs))

    func = relay.Function([data, weight, bn_gamma, bn_beta, bn_mean, bn_var], main_tuple)
    mod["main"] = func
    mod = relay.transform.InferType()(mod)
    return mod


if __name__ == '__main__':
    a = create_graph()
    print("================================")
    print("Annotated graph")
    print(a)
    print("================================")

    print("================================")
    print("Expected Graph")
    b = expected()
    print(b)
    print("================================")
    mod = tvm.IRModule()
    mod["main"] = create_graph()
    mod = transform.PartitionGraph()(mod)
    # fused_mod = transform.FuseOps(2)(mod)
    print("================================")
    print("After partition")
    print(mod)
    print("================================")
