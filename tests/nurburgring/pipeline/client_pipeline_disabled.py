import socket
import pickle
from warnings import formatwarning
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
parser.add_argument('--img_size', '-i', type=int, default=512, help='set image size')
parser.add_argument('--model', '-m', type=str, default='unet', help='name of model')
parser.add_argument('--target', '-t', type=str, default='llvm', help='name of taget')
parser.add_argument('--opt_level', '-o', type=int, default=2, help='set opt_level')
parser.add_argument('--ip', type=str, default='127.0.0.1', help='input ip of host')
parser.add_argument('--socket_size', type=int, default=1024*1024, help='socket data size')
parser.add_argument('--ntp_enable', type=int, default=0, help='ntp support')
parser.add_argument('--visualize', '-v', type=int, default=0, help='visualize option')
args = parser.parse_args()

def get_time(is_enabled):
    if is_enabled == 1:
        return g_ntp_client.request(ntp_time_server, version=3).tx_time
    elif is_enabled == 0:
        return time.time()
    else:
        return 0

def make_preprocess(model, im_sz):
    if model == 'unet':
        def preprocess(img):
            return cv2.resize(img[490:1800, 900:2850], (im_sz,im_sz)).astype(np.float32) / 255
        return preprocess
    elif model == 'resnet152':
        def preprocess(img):
            return cv2.resize(img, (im_sz, im_sz))
        return preprocess

def to_8bit(num):
    float16 = num.astype(np.float16) # Here's some data in an array
    float8s = float16.tobytes()[1::2]
    return float8s

def from_8bit(num):
    float16 = np.frombuffer(np.array(np.frombuffer(num, dtype='u1'), dtype='>u2').tobytes(), dtype='f2')
    return float16.astype(np.float32)

preprocess = make_preprocess(args.model, args.img_size)

# Initialize connect
HOST_IP = args.ip
PORT = 9998       
socket_size = args.socket_size

client_socket  = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
client_socket.connect((HOST_IP, PORT))

# target and dev set
if args.target == 'llvm':
    target = 'llvm'
    dev = tvm.cpu()
elif args.target == 'cuda':
    target = 'cuda'
    dev = tvm.cuda()
elif args.target == 'opencl':
    target = 'opencl'
    dev = tvm.opencl()

# Video Load
img_size = args.img_size 

org=(50,100)
font=cv2.FONT_HERSHEY_SIMPLEX

def send_data(socket, data):
    if data == None:
        send_msg = struct.pack('i', 0)
    else:
        # Packing data
        msg_body = pickle.dumps(data)
        total_send_msg_size = len(msg_body)
        send_msg = struct.pack('i', total_send_msg_size) + msg_body
    # Send object
    socket.sendall(send_msg)

def recv_data(socket, buf):
    # ######################################################
    # # Message Protocol
    # # | data_size (int) [4] byte | data [data_size] byte | 
    # ######################################################
    while len(buf) < 4:
        buf += socket.recv(4)

    # Get receive data size
    data_size_buf = buf[:4]
    data_size = struct.unpack('i', data_size_buf)[0]
    # Cut front
    buf = buf[4:]

    # Exit condition
    if data_size == 0:
        return None, b''  

    # Receive data object
    while len(buf) < data_size:
        buf += client_socket.recv(data_size - len(buf))

    data = pickle.loads(buf[:data_size])
    
    # Return data and remain buffer
    return data, buf[data_size:]

def post_processing(data, mask, visualize=False):
    img_in_rgb = data
    th = cv2.resize(cv2.threshold(np.squeeze(mask.transpose([0,2,3,1])), 0.5, 1, cv2.THRESH_BINARY)[-1], (img_size,img_size))
    img_in_rgb[th == 1] = [0, 0, 255]
    if visualize:
        cv2.imshow("received - client", img_in_rgb)
        cv2.waitKey(1)


def read_and_inference():
    # ##########################################################
    # Load models
    # ##########################################################
    model_path = "../src/model/{}_{}_full_{}_{}.so".format(args.model, args.target, args.img_size, args.opt_level)
    lib = tvm.runtime.load_module(model_path)
    partition_points = args.partition_points
    current_file_path = os.path.dirname(os.path.realpath(__file__)) + "/"

    model_input_indexs = []
    model_output_indexs = []
    model_graph_json_strs = []

    for i in range(len(partition_points) - 1):
        start_point = partition_points[i]
        end_point = partition_points[i + 1]
        with open(current_file_path + "../src/graph/{}_{}_{}_{}_{}-{}.json".format(args.model, args.target, args.img_size, args.opt_level, start_point, end_point), "r") as json_file:
            graph_json = json.load(json_file)
        input_indexs = graph_json['extra']["inputs"]
        output_indexs = graph_json['extra']["outputs"]
        
        model_input_indexs.append(input_indexs)
        model_output_indexs.append(output_indexs)
        del graph_json['extra']
        model_graph_json_strs.append(json.dumps(graph_json))

    param_path = "../src/model/{}_{}_full_{}_{}.params".format(args.model, args.target, args.img_size, args.opt_level)
    with open(param_path, "rb") as fi:
        loaded_params = bytearray(fi.read())

    models = []
    for graph_json_str in model_graph_json_strs:
        model = graph_executor.create(graph_json_str, lib, dev)
        model.load_params(loaded_params)
        models.append(model)
    # ##########################################################

    # Load video file
    cap = cv2.VideoCapture("../../../tvm-slicer/src/data/j_scan.mp4")
    
    model_outputs = {0 : 0}

    # Start loop
    fpss = []
    recv_buf = b''

    timer_client_total = 0
    timer_client_network = 0

    timer_client_total_start = time.time()
    # ##############################################
    # # Loop
    while (cap.isOpened()):
        # Timer read - start
        ret, frame = cap.read()                      
        try:
            frame = preprocess(frame)
        except:
            send_data(client_socket, None)
            break
        
        input_data = np.expand_dims(frame, 0).transpose([0, 3, 1, 2])

        model_outputs[0] = input_data

        output_indexs = []
        
        for model_index, [in_indexs, out_indexs, model] in enumerate(zip(model_input_indexs, model_output_indexs, models)):
            for input_index in in_indexs:
                model.set_input("input_{}".format(input_index), model_outputs[input_index])
            
            # run model
            model.run()
            # # SYNC
            # dev.sync()

            # TIMER
            timer_client_network_start = time.time()
            if len(output_indexs) != 0:
                packed_data = {}
                for idx in output_indexs:
                    packed_data[idx] = model_outputs[idx]
                send_data(client_socket, packed_data)

            # TIMER end
            timer_client_network += time.time() - timer_client_network_start

            # get output
            for i, output_index in enumerate(out_indexs):
                model_outputs[output_index] = model.get_output(i).numpy()

            output_indexs = out_indexs

        # TIMER
        timer_client_network_start = time.time()
        # Cloud - Send only case
        if len(models) == 0:
            output_indexs = [0]

        if len(output_indexs) != 0:
            packed_data = {}
            for idx in output_indexs:
                packed_data[idx] = model_outputs[idx]
            send_data(client_socket, packed_data)

        data, recv_buf = recv_data(client_socket, recv_buf)
        # TIMER end
        timer_client_network += time.time() - timer_client_network_start

        post_processing(frame, data[list(data.keys())[0]], visualize=args.visualize)     
    # ##############################################

    timer_client_total += time.time() - timer_client_total_start

    print("========================")
    print("timer_client_total", timer_client_total)
    print("timer_client_network", timer_client_network)
    print("timer_client_exclude network", timer_client_total - timer_client_network)
    print("========================")

    cv2.destroyAllWindows()
    client_socket.close()

if __name__ == '__main__':
    print(args.model, ", ", args.target, ", ", args.img_size, ", ", args.opt_level, ", ", 'partition points :', args.partition_points, sep='')
    p1 = Process(target=read_and_inference)
    p1.start() 
    stime = time.time()
    p1.join(); 
