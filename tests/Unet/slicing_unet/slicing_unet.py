from SlicingMachine import TVMSlicer
import tensorflow as tf
import tvm
import tvm.relay as relay
from tvm.contrib import graph_executor
import numpy as np
import os
import json
import pygraphviz as pgv

os.environ [ "TF_FORCE_GPU_ALLOW_GROWTH" ] = "true"

def show_graph(json_data, file_name=None):
    if type(json_data) == str:
        json_data = json.loads(json_data)
    A = pgv.AGraph(directed=True)
    for node_idx, node in enumerate(json_data['nodes']):
        for src in node['inputs']:
            A.add_edge(json_data['nodes'][src[0]]['name'] + '[{}]'.format(src[0]), node['name'] + '[{}]'.format(node_idx))
    if file_name:
        A.draw(file_name + '.png', format='png', prog='dot')

np.random.seed(12)
input_data = np.random.normal(0,1,(1,512,512,3)).astype(np.float32)

## import original and check output
# model_keras = tf.keras.models.load_model('./unet_512.h5')
model_keras = tf.keras.models.load_model('./unet_512_keras.h5')
# input_tensor = tf.convert_to_tensor(input_data)
out_keras = model_keras(input_data).numpy()
print(out_keras[0][0][:10].T)
print(out_keras.shape)

## tvm result
input_data = input_data.transpose([0, 3, 1, 2])
shape_dict = {"input_1": input_data.shape}
mod, params = relay.frontend.from_keras(model_keras, shape_dict)
# target = 'llvm'
target = 'cuda'

# dev = tvm.cpu()
dev = tvm.cuda()
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target, params=params)
    
show_graph(lib['get_graph_json'](), "unet_{}_lv_{}".format(target, 3))
with open("unet_{}_lv_{}.json".format(target, 3), "w") as json_file:
    json_file.write(lib['get_graph_json']())

model_tvm = graph_executor.GraphModule(lib["default"](dev))
model_tvm.set_input('input_1', input_data)
model_tvm.run()
out_tvm = model_tvm.get_output(0).asnumpy()

out_tvm = out_tvm.transpose([0, 2, 3, 1])
print(out_tvm[0][0][:10].T)
print(out_tvm.shape)
graph_json_raw = lib['get_graph_json']()
# graph_json_front, graph_json_back = TVMSlicer(lib['get_graph_json'](), 10).get_graph()
graph_json_front_info, graph_json_back_info = TVMSlicer(lib['get_graph_json'](), [[0,39],[39,111]]).get_graph()
graph_json_front, input_front, output_front = graph_json_front_info
graph_json_back, input_back, output_back = graph_json_back_info

print(input_front, input_back)
print(output_front, output_back)

# print(json.dumps(graph_json_front))
# print(json.dumps(graph_json_back))

with tvm.transform.PassContext(opt_level=3):
    lib = relay.build_graph(mod, target=target, target_host=None, params=params, mod_name="default", graph_config=json.dumps(graph_json_front))

model_tvm_front = graph_executor.GraphModule(lib["default"](dev))
model_tvm_front.set_input('input_0', input_data)
model_tvm_front.run()
print("FRONT END")
out_tvm_fronts = []
for i in range(len(output_front)):
    out_tvm_fronts.append(model_tvm_front.get_output(i).asnumpy())

# print(json.dumps(graph_json_back))
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build_graph(mod, target=target, target_host=None, params=params, mod_name="default", graph_config=json.dumps(graph_json_back))


model_tvm_back = graph_executor.GraphModule(lib["default"](dev))
for i, idx in enumerate(input_back):
    model_tvm_back.set_input('input_{}'.format(idx), out_tvm_fronts[i])
model_tvm_back.run()
out_tvm_back = model_tvm_back.get_output(0).asnumpy()

out_tvm_back = out_tvm_back.transpose([0, 2, 3, 1])
print(out_tvm_back.shape)
print(out_tvm_back[0][0][:10].T)

print(np.allclose(out_tvm, out_tvm_back))
