import socket
import pickle
# import cloudpickle as pickle
import numpy as np
import tvm
from tvm import te
import tvm.relay as relay
from tvm.contrib.download import download_testdata
from tensorflow import keras
from tvm.contrib import graph_executor
import json
import time
from PIL import Image
#import subprocess

HOST = '192.168.0.4'
#HOST = 'givenarc.iptime.org'
PORT = 9999        

marker = b"Q!W@E#R$"
end_marker = b"$R#E@W!Q"
socket_size = 1024 

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind((HOST, PORT))
target = 'cuda'
dev = tvm.cuda(0)
lib_path = '../src/model/'

while True:
    
    server_socket.listen()
    
    client_socket, addr = server_socket.accept()
    
    
    partition_point = int(str(client_socket.recv(socket_size).decode()))
   
    print("Run partition point at {}".format(partition_point))

    #full_lib_file = 'local_lib_full_{}.tar'.format(str(partition_point))
    #full_lib = tvm.runtime.load_module(lib_path + full_lib_file)
    #full_model = graph_executor.GraphModule(full_lib["default"](dev))
 
    back_lib_file = 'local_lib_back_{}.tar'.format(str(partition_point))
    back_lib = tvm.runtime.load_module(lib_path + back_lib_file)
    back_model = graph_executor.GraphModule(back_lib["default"](dev))
    
    #while True:
    #    data = []
    #    while True:
    #        packet = client_socket.recv(socket_size)
    #        if marker in packet:
    #            data.append(packet.split(marker)[0])
    #            break
    #        data.append(packet)
    #    byte_obj = b"".join(data)
    #    if end_marker in byte_obj:
    #        break
    #    front_out = pickle.loads(byte_obj)
    #    front_out = np.array(front_out)
    #
    #    input_data = tvm.nd.array(front_out)
    #    full_model.set_input('input_1', input_data)
    #    full_model.run()
    #    full_out = full_model.get_output(0).asnumpy()
    #    byte_obj = pickle.dumps(full_out)
    #    client_socket.sendall(byte_obj)
    #    client_socket.sendall(marker)

    #print("FULL FINISH")
    # 
    #del full_lib
    #del full_model
    #del data
    #del byte_obj

    #usage = []
    while True:
        data = []
        while True:
            packet = client_socket.recv(socket_size)
            if marker in packet:
                data.append(packet.split(marker)[0])
                break
            data.append(packet)
        byte_obj = b"".join(data)
        if end_marker in byte_obj:
            break
        #print(byte_obj)
        try:
            front_out = pickle.loads(byte_obj)
        except:
            break
        front_out = np.array(front_out)
        input_data = tvm.nd.array(front_out)
        print("error occured")
        packet = client_socket.recv(socket_size)

        back_model.set_input('input_1', input_data)
        back_model.run()
        #usage.append(int(subprocess.check_output("nvidia-smi | grep 3019MiB | grep -o -E '[0-9]+%' | tail -1", shell=True).decode().split('%')[0]))
        out_deploy = back_model.get_output(0).asnumpy()
        byte_obj = pickle.dumps(np.array(out_deploy))
        client_socket.sendall(byte_obj)
        client_socket.sendall(marker)
    #print(sum(usage) / len(usage), '%')
    client_socket.close()
    del back_lib
    del back_model
    del byte_obj

server_socket.close()