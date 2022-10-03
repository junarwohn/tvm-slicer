import re
import socket
import pickle
from tracemalloc import start
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
from multiprocessing import Process, Queue
import os

ntp_time_server = 'time.windows.com'               # NTP Server Domain Or IP 
ntp_time_server = 'time.google.com'               # NTP Server Domain Or IP 
g_ntp_client = ntplib.NTPClient() 

parser = ArgumentParser()
parser.add_argument('--start_point', '-s', type=int, default=0)
parser.add_argument('--end_point', '-e', type=int, default=-1)
parser.add_argument('--partition_points', '-p', nargs='+', type=str, default=[], help='set partition points')
parser.add_argument('--front', '-f', nargs='+', type=int, default=0, help='set front partition point')
parser.add_argument('--back', '-b', nargs='+', type=int, default=0, help='set back partition point')
parser.add_argument('--img_size', '-i', type=int, default=256, help='set image size')
parser.add_argument('--model', '-m', type=str, default='unet', help='name of model')
parser.add_argument('--target', '-t', type=str, default='cuda', help='name of taget')
parser.add_argument('--opt_level', '-o', type=int, default=3, help='set opt_level')
parser.add_argument('--ip', type=str, default='192.168.0.184', help='input ip of host')
parser.add_argument('--socket_size', type=int, default=1024*1024, help='socket data size')
parser.add_argument('--ntp_enable', type=int, default=0, help='ntp support')
parser.add_argument('--visualize', '-v', type=int, default=0, help='visualize option')
parser.add_argument('--model_config', '-c', nargs='+', type=int, default=0, help='set partition point')
parser.add_argument('--quantization_level', '-q', type=int, default=0, help='set quantization level')
args = parser.parse_args()

model_config = args.model_config
quantization_level = args.quantization_level

def get_time(is_enabled):
    if is_enabled == 1:
        return g_ntp_client.request(ntp_time_server, version=3).tx_time
    else:
        return time.time()

def to_8bit(num):
    float16 = num.astype(np.float16) # Here's some data in an array
    float8s = float16.tobytes()[1::2]
    return float8s

def from_8bit(num):
    float16 = np.frombuffer(np.array(np.frombuffer(num, dtype='u1'), dtype='>u2').tobytes(), dtype='f2')
    return float16.astype(np.float32)

def get_model_info(partition_points):
    model_input_indexs = []
    model_output_indexs = []
    model_graph_json_strs = []

    # Load front model json infos
    for i in range(len(partition_points) - 1):
        # start_point = partition_points[i]
        # end_point = partition_points[i + 1]
        start_points = [int(i) + 1 for i in partition_points[i].split(',')]
        end_points =  [int(i) for i in partition_points[i + 1].split(',')]

        # with open(current_file_path + "../src/graph/{}_{}_{}_{}_{}-{}.json".format(args.model, args.target, args.img_size, args.opt_level, start_point, end_point), "r") as json_file:
        with open("UNet_M[{}-{}-{}-{}]_Q[{}]_S[{}-{}].json".format(
            *model_config, 
            quantization_level, 
            "_".join(map(str,[i - 1 for i in start_points])), 
            "_".join(map(str, end_points))), "r") as json_file:
        # with open("unet_as_{}_{}_{}_{}_{}-{}.json".format(*model_config, start_point, end_point), "r") as json_file:
            graph_json = json.load(json_file)
        input_indexs = graph_json['extra']["inputs"]
        output_indexs = graph_json['extra']["outputs"]
        model_input_indexs.append(input_indexs)
        model_output_indexs.append(output_indexs)
        del graph_json['extra']
        model_graph_json_strs.append(json.dumps(graph_json))

    return model_input_indexs, model_output_indexs, model_graph_json_strs

# Initialize connect
HOST_IP = args.ip
PORT = 9998        
socket_size = args.socket_size

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind((HOST_IP, PORT))

server_socket.listen()
client_socket, addr = server_socket.accept()

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



if __name__ == '__main__':

    # Load lib
    lib_path = "UNet_M[{}-{}-{}-{}]_Q[{}]_full.so".format(*model_config, quantization_level)
    # lib_path = "../src/model/{}_{}_full_{}_{}.so".format(args.model, args.target, args.img_size, args.opt_level)
    lib = tvm.runtime.load_module(lib_path)

    # Load params
    params_path = "UNet_M[{}-{}-{}-{}]_Q[{}]_full.params".format(*model_config, quantization_level)
    # params_path = "../src/model/{}_{}_full_{}_{}.params".format(args.model, args.target, args.img_size, args.opt_level)
    with open(params_path, "rb") as fi:
        loaded_params = bytearray(fi.read())

    server_input_idxs, server_output_idxs, server_graph_json_strs = get_model_info(args.partition_points)

    # Assume that there is only one model    
    model = graph_executor.create(server_graph_json_strs[0], lib, dev)
    print("load params")
    model.load_params(loaded_params)

    print("ll")
    total_inputs = len(server_input_idxs[0])
    
    recv_msg = b''

    client_connection = True
    while True:
        # recv messages as the number of inputs
        # if recv the 0 size packet -> break the main loop
        recv_queue = {}
        while len(recv_queue.keys()) < total_inputs:
            try:
                while len(recv_msg) < 4:
                    recv_msg += client_socket.recv(4)
                msg_size_bytes = recv_msg[:4]
                recv_msg = recv_msg[4:]
                total_recv_msg_size = struct.unpack('i', msg_size_bytes)[0]
                
                # Recv end signal
                if total_recv_msg_size == 0:
                    # send the end signal : 0
                    send_msg = struct.pack('i', 0)
                    client_socket.sendall(send_msg)
                    client_connection = False
                    break

                while len(recv_msg) < total_recv_msg_size:
                    recv_msg += client_socket.recv(total_recv_msg_size)

                msg_data_bytes = recv_msg[:total_recv_msg_size]
                recv_msg = recv_msg[total_recv_msg_size:]
                data = pickle.loads(msg_data_bytes)
                for k in data:
                    recv_queue[k] = data[k]

            except:
                break
        
        if not client_connection:
            break

        # inference
        for key in recv_queue.keys():
            # print('input_{}'.format(key))
            # print(recv_queue[key].shape)
            model.set_input('input_{}'.format(key), recv_queue[key])
            
        model.run()

        send_queue = {}
        # get output from the server_output_idxs
        for i, out_idx in enumerate(server_output_idxs[0]):
            out = model.get_output(i).numpy()
            send_queue[out_idx] = out

        # send the output by wrapping by {idx: content}
        msg_body = pickle.dumps(send_queue)
        total_send_msg_size = len(msg_body)
    
        # print("Send msg:", total_send_msg_size)
        send_msg = struct.pack('i', total_send_msg_size) + msg_body

        # Send object
        client_socket.sendall(send_msg)

    client_socket.close()
    
