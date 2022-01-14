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
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument('--ip', type=str, default='192.168.0.184', help='input ip of host')
parser.add_argument('--device', type=str, default='cuda', help='type of devices [llvm, cuda]')
parser.add_argument('--socket_size', type=int, default=1024*1024, help='socket data size')

args = parser.parse_args()

# Model Load

if args.device == 'cuda':
    target = 'cuda'
    dev = tvm.cuda()
elif args.device == 'llvm':
    target = 'llvm'
    dev = tvm.cpu()
else:
    raise Exception("Wrong device")

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

HOST_IP = args.ip
PORT = 9998        
#socket_size = 16 * 1024 * 1024 
socket_size = args.socket_size

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind((HOST_IP, PORT))

server_socket.listen()
client_socket, addr = server_socket.accept()

print("Connection estabilished")
total_time_start = time.time()
inference_time = 0
network_time = 0
data_transmission_time = 0
while True:
    ins = []

    # Measuring time : Data Transmission
    time_sent = struct.unpack('d', client_socket.recv(8))[0]
    try:
        num_data = struct.unpack('i', client_socket.recv(4))[0]
        if num_data == 0:
            break
    except:
        break
    len_data = struct.unpack('i', client_socket.recv(4))[0]
    recv_msg = client_socket.recv(socket_size)
    while len(recv_msg) < len_data:
        packet = client_socket.recv(socket_size)
        recv_msg += packet

    for idx, shape in zip(input_info, shape_info):
        in_idx = struct.unpack('i', recv_msg[:4])[0]
        if idx != in_idx:
            raise Exception("Input not matched")
        msg_len = struct.unpack('i', recv_msg[4:8])[0]
        ins.append([idx, np.frombuffer(recv_msg[8:8+msg_len], np.float16).reshape(shape)])
        recv_msg = recv_msg[8+msg_len:]
    
    # Measuring time : Data Transmission end
    network_time += time.time() - time_sent

    inference_time_start = time.time()
    for idx, indata in ins:
        back_model.set_input("input_{}".format(idx), indata)

    back_model.run()
    out = back_model.get_output(0).asnumpy().astype(np.float32)
    inference_time += time.time() - inference_time_start
   
    # Time start
    time_sent = struct.pack('d', time.time())
    send_obj = out.astype(np.float16).tobytes()
    send_obj_len = len(send_obj)
    send_msg = time_sent + struct.pack('i', 0) + struct.pack('i', send_obj_len) + send_obj
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

total_time = time.time() - total_time_start
print("total time :", total_time)
print("inference time :", inference_time)
print("network time :", network_time)

client_socket.close()

server_socket.close()
