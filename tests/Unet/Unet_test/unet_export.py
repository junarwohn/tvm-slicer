import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from UNetKerasAS import UNet as UnetAS
import tensorflow as tf
import tvm
import tvm.relay as relay
from tvm.contrib import graph_executor
import numpy as np

model_config = [0,0,0,0]
model = UnetAS(3, 1, 64, model_config)
model.build(input_shape=(1,256, 256, 3))
model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4), loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['binary_crossentropy'])
model.save("unet_as_{}_{}_{}_{}.h5".format(*model_config))

img_size = 256

input_data = np.random.normal(0,1,(1,img_size,img_size,3)).astype(np.float32)
input_data = input_data.transpose([0, 3, 1, 2])
model = tf.keras.models.load_model("unet_as_{}_{}_{}_{}.h5".format(*model_config))
shape_dict = {"input_1": input_data.shape}
mod, params = relay.frontend.from_keras(model, shape_dict)
target = 'cuda'
dev = tvm.cuda()
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target, params=params)

lib.export_library("unet_as_{}_{}_{}_{}.so".format(*model_config))

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
