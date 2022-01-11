import socket
import pickle
# import cloudpickle as pickle
import numpy as np
import tvm
from tvm import te
import tvm.relay as relay
from tvm.contrib.download import download_testdata
from tvm.contrib import graph_executor
import json
import time
import sys
import cv2
import struct

# Model Load

target = 'cuda'
dev = tvm.cuda(0)
model_path = "../src/model/unet_tvm_back.so"
back_lib = tvm.runtime.load_module(model_path)
back_model = graph_executor.GraphModule(back_lib['default'](dev))

model_info_path = "../src/graph/graph_json_back.json"
with open(model_info_path, "r") as json_file:
    model_info = json.load(json_file)

input_info = model_info["extra"]["inputs"]
shape_info = model_info["attrs"]["shape"][1][:len(input_info)]
output_info = model_info["extra"]["outputs"]

print("Model Loaded")

# Initialize connect

#HOST = '192.168.0.184'
HOST = '192.168.0.190'
PORT = 9998        
#socket_size = 1024 * 1024 * 1024 
socket_size = 1024 

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind((HOST, PORT))

server_socket.listen()
client_socket, addr = server_socket.accept()

print("Connection estabilished")

while True:
    ins = []
    
    for idx, shape in zip(input_info, shape_info):
        in_idx = struct.unpack('i', client_socket.recv(4))[0]
        if idx != in_idx:
            raise Exception("Input not matched")
        msg_len = struct.unpack('i', client_socket.recv(4))[0]
        #print("receive", in_idx, msg_len)
        packet = client_socket.recv(socket_size)
        # packet = client_socket.recv()
        recv_msg = packet
        while len(recv_msg) < msg_len:
            # print(len(recv_msg))
            packet = client_socket.recv(socket_size)
            # packet = client_socket.recv()
            recv_msg += packet
        client_socket.sendall(struct.pack('i', 1))
        ins.append([idx, np.frombuffer(recv_msg, np.float32).reshape(shape)])

    for idx, indata in ins:
        back_model.set_input("input_{}".format(idx), indata)

    back_model.run()
    out = back_model.get_output(0).asnumpy().astype(np.float32)

    cv2.imshow("received - server", 255 * out.transpose([0,2,3,1])[0])
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    send_obj = out.tobytes()
    send_obj_len = len(send_obj)
    send_msg = struct.pack('i', 0) + struct.pack('i', send_obj_len) + send_obj
    client_socket.sendall(send_msg)

    # msg_len = struct.unpack('i', client_socket.recv(4))[0]
    # if msg_len == 0:
    #     break
    # packet = client_socket.recv(socket_size)
    # recv_msg = packet
    # while len(recv_msg)  < msg_len:
    #     packet = client_socket.recv(socket_size)
    #     recv_msg += packet
    # recv_data = np.frombuffer(recv_msg, np.uint8).reshape(1,3,512,512)
    # cv2.imshow("received - server", recv_data.transpose([0,2,3,1])[0])
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break
    # send_obj = recv_data.tobytes()
    # send_obj_len = len(send_obj)
    # send_msg = struct.pack('i', send_obj_len) + send_obj
    # client_socket.sendall(send_msg)

client_socket.close()

server_socket.close()
