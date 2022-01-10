import socket
import pickle
# import cloudpickle as pickle
import numpy as np
import tvm
from tvm import te
import tvm.relay as relay
from tvm.contrib.download import download_testdata
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import preprocess_input
from tvm.contrib import graph_executor
import json
import time
from PIL import Image
from matplotlib import pyplot as plt
import sys
try:
    partition_point = int(sys.argv[1])
except:
    partition_point = 72


## Image Load
img_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
img_path = download_testdata(img_url, "cat.png", module="data")
img = Image.open(img_path).resize((224, 224))
# input preprocess
img_data = np.array(img)[np.newaxis, :].astype("float32")
img_data = preprocess_input(img_data).transpose([0, 3, 1, 2])

# Initialize connect

HOST = '192.168.0.4'  
#HOST = 'givenarc.iptime.org'
PORT = 9999       

marker = b"Q!W@E#R$"
end_marker = b"$R#E@W!Q"
socket_size = 1024 
client_socket  = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
client_socket.connect((HOST, PORT))

# Send partition point

msg = str(partition_point)
client_socket.sendall(msg.encode())
 
# Load the model
batch_size = 1000  
print("run batch size {}".format(str(batch_size)))
target = "cuda"

dev = tvm.cuda(0)

lib_path = './model_build/'
#full_lib_file = 'local_lib_full.tar'
front_lib_file = 'local_lib_front_{}.tar'.format(str(partition_point))
#full_lib = tvm.runtime.load_module(lib_path + full_lib_file)
front_lib = tvm.runtime.load_module(lib_path + front_lib_file)

#full_model = graph_executor.GraphModule(full_lib["default"](dev))
front_model = graph_executor.GraphModule(front_lib["default"](dev))

# Run front in local, back in remote
partition_time_start = time.time()
front_time = 0
back_time = 0

client_socket.settimeout(0.8)
partition_start = time.time()
# while run_time < batch_size:
for run_time in range(batch_size):

    input_data = tvm.nd.array(img_data)
    front_model.set_input('input_1', input_data)
    front_model.run()
    out_deploy = front_model.get_output(0).asnumpy()


    byte_obj = pickle.dumps(np.array(out_deploy))
    for k in range(10):
        try:
            #print(byte_obj)
            #print("+==================+")
            client_socket.sendall(byte_obj)
            client_socket.sendall(marker)
            data = []
            while True:
                packet = client_socket.recv(socket_size)
                if marker in packet:
                    data.append(packet.split(marker)[0])
                    break
                data.append(packet)
        except socket.timeout:
            print("timeout")
            continue
        break
    byte_obj = b"".join(data)
    back_out = pickle.loads(byte_obj)
    back_out = np.array(back_out)


partition_time = time.time() - partition_start 

client_socket.sendall(end_marker)
client_socket.sendall(marker)
    
top1_tvm = np.argmax(back_out)# print(out_deploy[0])

client_socket.close()

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


# INPUT LOCAL FULL TIME
#local_full_time = 0.0000000000
#top1_full_local = np.argmax(full_out)
#top1_full_remote = np.argmax(remote_full_out)
top1_partition = np.argmax(back_out)
print(time.strftime('%c', time.localtime(time.time())))
print("===============================================")
print("Partition Point {}".format(partition_point))
#print("full model  (local)     : {:.8f}s".format(local_full_time))
#print("full model  (remote)    : {:.8f}s".format(remote_full_time))
print("Parition Point: {:.8f}s".format(partition_time))
#print("Time local - partition  : {:.8f}s".format(local_full_time - partition_time))
#print("Time remote - partition : {:.8f}s".format(remote_full_time - partition_time))
print("===============================================")
#print("Is same result ?   : {}".format(np.array_equal(full_out, back_out)))
print("===============================================")
#print("Local Full Model top-1 id: {}, class name: {}".format(top1_full_local, synset[top1_full_local]))
#print("Remote Full Model top-1 id: {}, class name: {}".format(top1_full_remote, synset[top1_full_remote]))
print("Partition  top-1 id: {}, class name: {}".format(top1_partition, synset[top1_partition]))
print("===============================================")
print()
print()
