from http import client
import re
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
import ntplib 

ntp_time_server = 'time.windows.com'               # NTP Server Domain Or IP 
ntp_time_server = 'time.google.com'               # NTP Server Domain Or IP 
g_ntp_client = ntplib.NTPClient() 
#response = c.eequest(timeServer, version=3) 

parser = ArgumentParser()
parser.add_argument('--start_point', '-s', type=int, default=0)
parser.add_argument('--end_point', '-e', type=int, default=-1)
parser.add_argument('--partition_point', '-p', type=int, default=0, help='set partition point')
parser.add_argument('--img_size', '-i', type=int, default=512, help='set image size')
parser.add_argument('--model', '-m', type=str, default='unet', help='name of model')
parser.add_argument('--target', '-t', type=str, default='llvm', help='name of taget')
parser.add_argument('--opt_level', '-o', type=int, default=2, help='set opt_level')
parser.add_argument('--ip', type=str, default='127.0.0.1', help='input ip of host')
parser.add_argument('--socket_size', type=int, default=1024*1024, help='socket data size')
parser.add_argument('--ntp_enable', type=int, default=0, help='ntp support')

args = parser.parse_args()

def get_time(is_enabled):
    if is_enabled == 1:
        return g_ntp_client.request(ntp_time_server, version=3).tx_time
    else:
        return time.time()

# Model load

if args.target == 'llvm':
    target = 'llvm'
    dev = tvm.cpu()
elif args.target == 'cuda':
    target = 'cuda'
    dev = tvm.cuda()
elif args.target == 'opencl':
    target = 'opencl'
    dev = tvm.opencl()


model_path = "../src/model/{}_{}_back_{}_{}_{}.so".format(args.model, args.target, args.img_size, args.opt_level, args.partition_point)
back_lib = tvm.runtime.load_module(model_path)
back_model = graph_executor.GraphModule(back_lib['default'](dev))

model_info_path = "../src/graph/{}_{}_back_{}_{}_{}.json".format(args.model, args.target, args.img_size, args.opt_level, args.partition_point)

with open(model_info_path, "r") as json_file:
    model_info = json.load(json_file)

input_info = model_info["extra"]["inputs"]
shape_info = model_info["attrs"]["shape"][1][:len(input_info)]
output_info = model_info["extra"]["outputs"]

#print("Model Loaded")

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

# TODO check output size and send
# shape = (4,)
# total_output_num = len(model_info['heads'])
output_shapes = b''
for idxs in model_info['heads']:
    print(model_info["attrs"]["shape"][1][idxs[0]])
    output_shapes += struct.pack('i', len(model_info["attrs"]["shape"][1][idxs[0]])) + np.array(model_info["attrs"]["shape"][1][idxs[0]]).tobytes()

send_bytes = struct.pack('i', len(output_shapes)) + output_shapes
client_socket.sendall(send_bytes)
##

# timer INIT
timer_inference = 0
timer_total = 0
timer_exclude_network = 0

timer_toal_start = time.time()

recv_msg = b''
while True:
    try:
        while len(recv_msg) < 4:
            recv_msg += client_socket.recv(4)
        total_recv_msg_size = struct.unpack('i', recv_msg[:4])[0]
        recv_msg = recv_msg[4:]
        # total_recv_msg_size = struct.unpack('i', client_socket.recv(4))[0]
        if total_recv_msg_size == 0:
            client_socket.sendall(struct.pack('i', 0))
            break
    except:
        break
    while len(recv_msg) < total_recv_msg_size:
        # packet = client_socket.recvall()
        recv_msg += client_socket.recv(total_recv_msg_size)
    
    ### TIME_CHECK : UNPACK 
    ins = []
    for idx, shape in zip(input_info, shape_info):
        n,c,h,w = shape 
        msg_len = 4 * n * c * h * w
        ins.append([idx, np.frombuffer(recv_msg[:msg_len], np.float32).reshape(tuple(shape))])
        recv_msg = recv_msg[msg_len:]

    timer_exclude_network_start = time.time()

    timer_inference_start = time.time()

    for idx, indata in ins:
        back_model.set_input("input_{}".format(idx), indata)
    back_model.run()
    out = back_model.get_output(0).asnumpy().astype(np.float32)

    timer_inference += time.time() - timer_inference_start

    timer_exclude_network += time.time() - timer_exclude_network_start

    send_obj = out.tobytes()
    total_send_msg_size = len(send_obj)
    send_msg = struct.pack('i', total_send_msg_size) + send_obj

    client_socket.sendall(send_msg)

timer_total = time.time() - timer_toal_start
timer_network = timer_total - timer_exclude_network


print("total time :", timer_total)
print("inference time :", timer_inference)
print("exclude network time :", timer_exclude_network)
print("network time :", timer_network)

print("data receive size :", total_recv_msg_size)
print("data send size :", total_send_msg_size)

client_socket.close()
server_socket.close()
