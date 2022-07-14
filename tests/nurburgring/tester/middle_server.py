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
parser.add_argument('--partition_points', '-p', nargs='+', type=int, default=0, help='set partition point')
parser.add_argument('--front', '-f', nargs='+', type=int, default=0, help='set front partition point')
parser.add_argument('--back', '-b', nargs='+', type=int, default=0, help='set back partition point')
parser.add_argument('--img_size', '-i', type=int, default=512, help='set image size')
parser.add_argument('--model', '-m', type=str, default='unet', help='name of model')
parser.add_argument('--target', '-t', type=str, default='cuda', help='name of taget')
parser.add_argument('--opt_level', '-o', type=int, default=3, help='set opt_level')
parser.add_argument('--ip', type=str, default='192.168.0.184', help='input ip of host')
parser.add_argument('--socket_size', type=int, default=1024*1024, help='socket data size')
parser.add_argument('--ntp_enable', type=int, default=0, help='ntp support')
parser.add_argument('--visualize', '-v', type=int, default=0, help='visualize option')
args = parser.parse_args()


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
        start_point = partition_points[i]
        end_point = partition_points[i + 1]
        current_file_path = os.path.dirname(os.path.realpath(__file__)) + "/"
        with open(current_file_path + "../src/graph/{}_{}_{}_{}_{}-{}.json".format(args.model, args.target, args.img_size, args.opt_level, start_point, end_point), "r") as json_file:
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


def recv_data(recv_queue):
    recv_msg = b''
    while True:
        try:
            while len(recv_msg) < 4:
                recv_msg += client_socket.recv(4)
            msg_size_bytes = recv_msg[:4]
            recv_msg = recv_msg[4:]
            total_recv_msg_size = struct.unpack('i', msg_size_bytes)[0]
            if total_recv_msg_size == 0:
                recv_queue.put({-1 : -1})
                break
        except:
            break
 
        if total_recv_msg_size == 0:
            break

        # Format : {index : data, ...}
        while len(recv_msg) < total_recv_msg_size:
            recv_msg += client_socket.recv(total_recv_msg_size)
        
        msg_data_bytes = recv_msg[:total_recv_msg_size]
        recv_msg = recv_msg[total_recv_msg_size:]
        data = pickle.loads(msg_data_bytes)
        recv_queue.put(data)


def inference(recv_queue, send_queue):
    # Load lib
    lib_path = "../src/model/{}_{}_full_{}_{}.so".format(args.model, args.target, args.img_size, args.opt_level)
    lib = tvm.runtime.load_module(lib_path)

    # Load params
    params_path = "../src/model/{}_{}_full_{}_{}.params".format(args.model, args.target, args.img_size, args.opt_level)
    with open(params_path, "rb") as fi:
        loaded_params = bytearray(fi.read())

    server_input_idxs, server_output_idxs, server_graph_json_strs = get_model_info(args.partition_points)

    # create graph_executor
    # graph_json_str = json.dumps(server_graph_json_strs[0])
    model = graph_executor.create(server_graph_json_strs[0], lib, dev)
    model.load_params(loaded_params)

    # timer INIT
    timer_inference = 0
    timer_total = 0
    timer_exclude_network = 0

    timer_toal_start = time.time()
    recv_msg = b''

    total_result = []
        

    timer_model = 0
    total_inputs = len(server_input_idxs)
    recv_input_cnt = 0
    print("All Loaded")
    end_flag = False
    while True:
        if not recv_queue.empty():
            recv_input_cnt = 0
            # exit codition : {-1 : -1}
            while recv_input_cnt < total_inputs:
                # Get data
                data = recv_queue.get()
                if -1 in data.keys():
                    while recv_queue.qsize() != 0:
                        print("End inference")
                        # Clean recv_queue
                        recv_queue.get()
                    send_queue.put({-1 : -1})
                    end_flag = True
                    break

                for key in data.keys():
                    # print('input_{}'.format(key))
                    model.set_input('input_{}'.format(key), data[key])
                    recv_input_cnt += 1

            if end_flag:
                break

            time_start = time.time()
            model.run()
            timer_model += time.time() - time_start

            for i, out_idx in enumerate(server_output_idxs[0]):
                out = model.get_output(i).numpy()
                send_queue.put({out_idx : out})

# send in dict
def send_data(send_queue):
    while True:
        if not send_queue.empty():
            # Get data
            data = send_queue.get()

            # exit codition : {-1 : -1}
            if -1 in data.keys():
                while send_queue.qsize() != 0:
                    print("Exit Condition")
                    # Clean send_queue
                    send_queue.get()
                send_msg = struct.pack('i', 0)
                client_socket.sendall(send_msg)
                break
            # Packing data
            msg_body = pickle.dumps(data)
            total_send_msg_size = len(msg_body)
            # print("Send msg:", total_send_msg_size)
            send_msg = struct.pack('i', total_send_msg_size) + msg_body

            # Send object
            client_socket.sendall(send_msg)
        
    # Exit
    client_socket.close()
    print('send_img End')

if __name__ == '__main__':
    recv_queue = Queue()
    send_queue = Queue()

    recv_process = Process(target=recv_data, args=(recv_queue, ))
    main_process = Process(target=inference, args=(recv_queue, send_queue))
    send_process = Process(target=send_data, args=(send_queue,))

    recv_process.start()
    main_process.start() 
    send_process.start()

    # Timer check
    stime = time.time()

    recv_process.join()
    main_process.join(); 
    send_process.join(); 

    print(time.time() - stime)
