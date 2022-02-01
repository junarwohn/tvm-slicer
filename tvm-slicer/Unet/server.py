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
import ntplib
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
parser.add_argument('--ip', type=str, default='192.168.0.184', help='input ip of host')
parser.add_argument('--socket_size', type=int, default=1024*1024, help='socket data size')
parser.add_argument('--ntp_enable', type=int, default=1, help='ntp support')

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


model_path = "./src/model/{}_{}_back_{}_{}_{}.so".format(args.model, args.target, args.img_size, args.opt_level, args.partition_point)
back_lib = tvm.runtime.load_module(model_path)
back_model = graph_executor.GraphModule(back_lib['default'](dev))

model_info_path = "./src/graph/{}_{}_back_{}_{}_{}.json".format(args.model, args.target, args.img_size, args.opt_level, args.partition_point)

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
base_time = time.time()
#print(base_time)

#print("Connection estabilished")
total_time_start = time.time()
inference_time = 0
network_time = 0
data_transmission_time = 0

# TIME_CHECK INIT

time_checker = {
        'READ' : 0,
        'SET_INPUT' : 0,
        'RUN_MODEL': 0,
        'GET_OUTPUT' : 0,
        'PACK' : 0,
        'UNPACK' : 0,
        'VISUALIZE' : 0
}

# timer INIT
timer_READ = 0
timer_SET_INPUT = 0
timer_RUN_MODEL = 0
timer_GET_OUTPUT = 0
timer_ASNUMPY = 0
timer_VISUALIZE = 0


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
    
    ### TIME_CHECK : UNPACK 
    time_unpack_start = get_time(args.ntp_enable)

    for idx, shape in zip(input_info, shape_info):
        in_idx = struct.unpack('i', recv_msg[:4])[0]
        if idx != in_idx:
            raise Exception("Input not matched")
        msg_len = struct.unpack('i', recv_msg[4:8])[0]
        ins.append([idx, np.frombuffer(recv_msg[8:8+msg_len], np.float32).reshape(shape)])
        recv_msg = recv_msg[8+msg_len:]

    ### TIME_CHECK : UNPACK END
    time_checker['UNPACK'] += get_time(args.ntp_enable) - time_unpack_start

 
    # Measuring time : Data Transmission end
    network_time += get_time(args.ntp_enable) - time_sent 

    inference_time_start = time.time()

    ### TIME_CHECK : SET_INPUT
    time_set_input_start = get_time(args.ntp_enable)

    for idx, indata in ins:
        back_model.set_input("input_{}".format(idx), indata)

    ### TIME_CHECK : SET_INPUT DONE
    time_checker['SET_INPUT'] += get_time(args.ntp_enable) - time_set_input_start 


    ### TIME_CHECK : RUN_MODEL
    time_run_model_start = get_time(args.ntp_enable)
    back_model.run()

    ### TIME_CHECK : RUN_MODEL DONE
    time_checker['RUN_MODEL'] += get_time(args.ntp_enable) - time_run_model_start 


    ### TIME_CHECK : GET_OUTPUT
    time_get_output_start = get_time(args.ntp_enable)

    out = back_model.get_output(0).asnumpy().astype(np.float32)

    ### TIME_CHECK : GET_OUTPUT
    time_checker['GET_OUTPUT'] += get_time(args.ntp_enable) - time_get_output_start

    inference_time += time.time() - inference_time_start
   
    # Time start
    ### TIME_CHECK : PACK
    time_pack_start = get_time(args.ntp_enable)

    time_sent = struct.pack('d', get_time(args.ntp_enable))
    send_obj = out.tobytes()
    send_obj_len = len(send_obj)
    send_msg = time_sent + struct.pack('i', 0) + struct.pack('i', send_obj_len) + send_obj
    ### TIME_CHECK : PACK DONE
    time_checker['PACK'] += get_time(args.ntp_enable) - time_pack_start

    client_socket.sendall(send_msg)

total_time = time.time() - total_time_start
print("total time :", total_time)
print("inference time :", inference_time)
print("network time :", network_time)
print("number of input :", len(input_info))
print("index of input :", input_info[0])
print("number of output :", len(output_info))
print("index of output :", output_info[0])
print("data input size :", len_data)
print("data receive size :", len_data)
print("data send size :", send_obj_len)
total_time_checker = 0
for key in time_checker:
    print(key, ':', time_checker[key])
    total_time_checker += time_checker[key]
print("total_time_checker :", total_time_checker)

client_socket.close()
server_socket.close()
