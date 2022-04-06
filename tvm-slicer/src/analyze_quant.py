import tvm
import tensorflow as tf
import numpy as np
import tvm.relay as relay
import pygraphviz as pgv
import json

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

# model_keras = tf.keras.models.load_model('./very_simple_model.h5')
model_keras = tf.keras.models.load_model('./model/resnet152_224.h5')

input_data = np.random.normal(0,1,(1,224,224,3)).astype(np.float32)
input_data = input_data.transpose([0, 3, 1, 2])
shape_dict = {"input_1": input_data.shape}
mod, params = relay.frontend.from_keras(model_keras, shape_dict)

# print(params['_param_8'])

# print(mod)

# target = 'cuda'
# dev = tvm.cuda()
target = 'llvm'
dev = tvm.cpu()
# print(mod.astext())
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target, params=params)


# print("===before quantize===")
# print(lib.ir_mod.astext(show_meta_data=False))
# print("===               ===")


# with open("./relay_mod_save.json", "w") as json_file:
#     json_file.write(tvm.ir.save_json(mod))

# with open("./relay_param_save.json", "wb") as json_file:
#     json_file.write(tvm.runtime.save_param_dict(params))

# # print(lib['get_graph_json']())
show_graph(lib['get_graph_json'](), "resnet_vanila")

with relay.quantize.qconfig(calibrate_mode="global_scale", global_scale=8.0):
    mod = relay.quantize.quantize(mod, params)

# print(mod.astext())

with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target, params=params)
# print("===after  quantize===")
# print(lib.ir_mod.astext(show_meta_data=False))
# print("===               ===")
show_graph(lib['get_graph_json'](), "resnet_quant")


# # with tvm.transform.PassContext(opt_level=3):
#     # lib = relay.build(mod, target, params=params)
#     # print(lib['get_graph_json']())
# # print("===before quantize===")
# # print(mod)
# # print("===               ===")

# # with relay.quantize.qconfig(calibrate_mode="global_scale", global_scale=8.0):
# #     mod = relay.quantize.quantize(mod, params)
    

# # with tvm.transform.PassContext(opt_level=0):
# #     lib = relay.build(mod, target, params=params)

# # print("===after  quantize===")
# # print(mod)
# # print("===               ===")
# # print(lib['get_graph_json']())
