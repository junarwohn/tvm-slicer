import numpy as np
import tvm
from tvm.contrib import graph_executor

target = 'cuda'
dev = tvm.cuda()
img_size = 256
model_config = [0,0,2,0]
model_path = "unet_as_{}_{}_{}_{}.so".format(*model_config)
lib = tvm.runtime.load_module(model_path)
model = graph_executor.GraphModule(lib['default'](dev))

total_frames = 100
indata = tvm.nd.array(np.random.normal(0,1,(1,3,img_size, img_size)).astype('float32'), device=dev)

print(
    'set_input (ms) :', 
    1000 * model.module.time_evaluator(func_name='set_input', dev=dev, number=total_frames)('input_1', indata).results[0])
print(
    'run (ms) :', 
    1000 * model.module.time_evaluator(func_name='run', dev=dev, number=total_frames)().results[0])
