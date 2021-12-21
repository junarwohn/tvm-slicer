import time
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
from PIL import Image
import tvm
import tvm.relay as relay
from tvm import rpc
from tvm.contrib import utils
from tvm.contrib.download import download_testdata
from tvm.contrib import graph_executor

try:
    partition_point = int(sys.argv[1])
except:
    partition_point = 72
# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession

# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)


# Load Prerequisites
if tuple(keras.__version__.split(".")) < ("2", "4", "0"):
    weights_url = "".join(
        [
            "https://github.com/fchollet/deep-learning-models/releases/",
            "download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5",
        ]
    )
    weights_file = "resnet50_keras_old.h5"
else:
    weights_url = "".join(
        [
            " https://storage.googleapis.com/tensorflow/keras-applications/",
            "resnet/resnet50_weights_tf_dim_ordering_tf_kernels.h5",
        ]
    )
    weights_file = "resnet50_keras_new.h5"



weights_path = download_testdata(weights_url, weights_file, module="keras")
keras_resnet50 = keras.applications.resnet50.ResNet50(
    include_top=True, weights=None, input_shape=(224, 224, 3), classes=1000
)
keras_resnet50.load_weights(weights_path)


img_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
img_path = download_testdata(img_url, "cat.png", module="data")
img = Image.open(img_path).resize((224, 224))
# input preprocess
data = np.array(img)[np.newaxis, :].astype("float32")
data = preprocess_input(data).transpose([0, 3, 1, 2])

shape_dict = {"input_1": data.shape}
mod, params = relay.frontend.from_keras(keras_resnet50, shape_dict)

target = 'cuda'

with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target, params=params)
    front_lib, back_lib = relay.partition_graph(mod, [target, target], params=params, graph_config=lib.get_graph_json(), partition_point=partition_point)

local_dev = tvm.gpu()

front_model = graph_executor.GraphModule(front_lib["default"](local_dev))

back_model = graph_executor.GraphModule(back_lib["default"](local_dev))


time_start = time.time()

for i in range(1):
    # Load Data
    # img = Image.open(img_path).resize((224, 224))
    # data = np.array(img)[np.newaxis, :].astype("float32")
    # data = preprocess_input(data).transpose([0, 3, 1, 2])
    input_data = tvm.nd.array(data)
    
    # front_model = graph_executor.GraphModule(front_lib["default"](local_dev))

    front_model.set_input('input_1', input_data)
    front_model.run()
    front_out = front_model.get_output(0).asnumpy()
    input_data = tvm.nd.array(front_out)

    # back_model = graph_executor.GraphModule(back_lib["default"](local_dev))
    back_model.set_input('input_1', input_data)
    back_model.run()
    back_out = back_model.get_output(0).asnumpy()

print("input_data", input_data)
print("front_out", front_out)
print("back_out", back_out)

np.save("./front_out_local.npy", front_out)
np.save("./back_out.npy", back_out)

total_time = time.time() - time_start
top1_tvm = np.argmax(back_out)# print(out_deploy[0])
print("total time : {}".format(total_time))

synset_url = "".join(
    [
        "https://gist.githubusercontent.com/zhreshold/",
        "4d0b62f3d01426887599d4f7ede23ee5/raw/",
        "596b27d23537e5a1b5751d2b0481ef172f58b539/",
        "imagenet1000_clsid_to_human.txt",
    ]
)
synset_name = "imagenet1000_clsid_to_human.txt"
synset_path = download_testdata(synset_url, synset_name, module="data")
with open(synset_path) as f:
    synset = eval(f.read())
print("Relay top-1 id: {}, class name: {}".format(top1_tvm, synset[top1_tvm]))