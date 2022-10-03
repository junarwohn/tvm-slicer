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
from multiprocessing import Process, Queue
import os

ntp_time_server = 'time.windows.com'               # NTP Server Domain Or IP 
ntp_time_server = 'time.google.com'               # NTP Server Domain Or IP 
g_ntp_client = ntplib.NTPClient() 

parser = ArgumentParser()
parser.add_argument('--start_point', '-s', type=int, default=0)
parser.add_argument('--end_point', '-e', type=int, default=-1)
parser.add_argument('--partition_points', '-p', nargs='+', type=str, default=[], help='set partition points')
parser.add_argument('--front', '-f', nargs='+', type=str, default=[], help='set front partition point')
parser.add_argument('--back', '-b', nargs='+', type=str, default=[], help='set back partition point')
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
parser.add_argument('--jetson', '-j', type=int, default=0, help='jetson')
args = parser.parse_args()

def make_preprocess(model, im_sz):
    if model == 'unet':
        def preprocess(img):
            return cv2.resize(img[490:1800, 900:2850], (im_sz,im_sz)).astype(np.float32) / 255
        return preprocess
    elif model == 'resnet152':
        def preprocess(img):
            return cv2.resize(img, (im_sz, im_sz))
        return preprocess

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

model_config = args.model_config
quantization_level = args.quantization_level
is_jetson = args.jetson

def load_data():
    cap = cv2.VideoCapture("../../../tvm-slicer/src/data/j_scan.mp4")
    data_queue = []
    while (cap.isOpened()):
        ret, frame = cap.read()
        try:
            frame = preprocess(frame)    
        except:
            data_queue.append([])
            # frame_queue.put({-1:-1})
            # result_queue.put({-1:-1})
            break
        data_queue.append(frame)
    cap.release()
    return data_queue

def get_model_info(partition_points):
    model_input_indexs = []
    model_output_indexs = []
    model_graph_json_strs = []

    # If there is no model to be executed
    if len(partition_points) == 1:
        partition_points = list(map(int, partition_points))
        return [partition_points], [partition_points], []

    # Load front model json infos
    for i in range(len(partition_points) - 1):
        # start_point = partition_points[i]
        # end_point = partition_points[i + 1]
        start_points = [int(i) for i in partition_points[i].split(',')]
        end_points =  [int(i) for i in partition_points[i + 1].split(',')]
        # print(start_points, end_points)
        # current_file_path = os.path.dirname(os.path.realpath(__file__)) + "/"
        
        if is_jetson == 1:
            json_format = "UNet_M[{}-{}-{}-{}]_Q[{}]_S[{}-{}]_jetson.json"
        else:
            json_format = "UNet_M[{}-{}-{}-{}]_Q[{}]_S[{}-{}].json"

        with open(json_format.format(
            *model_config, 
            quantization_level, 
            "_".join(map(str,[i for i in start_points])), 
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

if __name__ == '__main__':
    print("------------------------")
    print(args.model, ", ", args.target, ", ", args.img_size, ", ", args.opt_level, ", ", 'partition points :', args.front, args.back, sep='')

    # Load video
    data_queue = load_data()

    # Load model
    points_front_model = args.front
    points_back_model = args.back
    points_server_model = [points_front_model[-1], points_back_model[0]]

    front_input_idxs, front_output_idxs, front_graph_json_strs = get_model_info(points_front_model)
    server_input_idxs, server_output_idxs, _ = get_model_info(points_server_model)
    back_input_idxs, back_output_idxs, back_graph_json_strs = get_model_info(points_back_model)

    total_front_output_idxs = []
    for i in front_output_idxs:
        total_front_output_idxs += i

    total_server_input_idxs = []
    for i in server_input_idxs:
        total_server_input_idxs += i

    total_back_input_idxs = []
    for i in back_input_idxs:
        total_back_input_idxs += i

    total_front_output_idxs = []
    for i in front_output_idxs:
        total_front_output_idxs += i

    total_server_output_idxs = []
    for i in server_output_idxs:
        total_server_output_idxs += i

    total_back_input_idxs = []
    for i in back_input_idxs:
        total_back_input_idxs += i

    # Obvious, but for safety
    # if not np.equal(np.setdiff1d(total_front_output_idxs, total_back_input_idxs), total_server_input_idxs):
    #     print(total_front_output_idxs, total_back_input_idxs, total_server_input_idxs)
    #     print("Error setting dependencies. It may be occured wrong slicing")
    #     exit()

    send_queue_idxs = total_server_input_idxs
    pass_queue_idxs = np.intersect1d(total_front_output_idxs, total_back_input_idxs)
    recv_queue_idxs = np.intersect1d(total_server_output_idxs, total_back_input_idxs)
    # print("###################")
    # print(total_front_output_idxs, total_server_output_idxs, total_back_input_idxs)
    # print(send_queue_idxs, pass_queue_idxs, recv_queue_idxs)
    # print("###################")

    # Load models
    if is_jetson == 1:
        model_format = "UNet_M[{}-{}-{}-{}]_Q[{}]_full_jetson.so"
    else:
        model_format = "UNet_M[{}-{}-{}-{}]_Q[{}]_full.so"
    
    model_path = model_format.format(*model_config, quantization_level)
    lib = tvm.runtime.load_module(model_path)

    if is_jetson == 1:
        param_format = "UNet_M[{}-{}-{}-{}]_Q[{}]_full_jetson.params"
    else:
        param_format = "UNet_M[{}-{}-{}-{}]_Q[{}]_full.params"

    param_path = param_format.format(*model_config, quantization_level)
    with open(param_path, "rb") as fi:
        loaded_params = bytearray(fi.read())

    front_models = []
    for graph_json_str in front_graph_json_strs:
        model = graph_executor.create(graph_json_str, lib, dev)
        model.load_params(loaded_params)
        front_models.append(model)

    back_models = []
    for graph_json_str in back_graph_json_strs:
        model = graph_executor.create(graph_json_str, lib, dev)
        model.load_params(loaded_params)
        back_models.append(model)

    in_data = {0 : 0}
    pass_queue = {}
    recv_msg = b''
    # Load network connection
    stime = time.time()
    for frame in data_queue:
        in_data = {}
        # Data preprocessing
        if len(frame) == 0:
            # Exit condition
            send_msg = struct.pack('i', 0)
            client_socket.sendall(send_msg)
            break
        
        input_data = np.expand_dims(frame, 0).transpose([0, 3, 1, 2])
        in_data[0] = input_data

        # Front inference
        for in_indexs, out_indexs, model in zip(front_input_idxs, front_output_idxs, front_models):
            # set input
            for input_index in in_indexs:
                model.set_input("input_{}".format(input_index), in_data[input_index])
            # run model
            model.run()
            for i, output_index in enumerate(out_indexs):
                in_data[output_index] = model.get_output(i).numpy()
        # Send
        for output_idxs in front_output_idxs:
            for output_idx in output_idxs:
                if output_idx in send_queue_idxs:
                    msg_body = pickle.dumps({output_idx : in_data[output_idx]})
                    total_send_msg_size = len(msg_body)
                    send_msg = struct.pack('i', total_send_msg_size) + msg_body
                    client_socket.sendall(send_msg)
                if output_idx in pass_queue_idxs:
                    pass_queue[output_idx] = in_data[output_idx]
        # Recv
        recv_queue = {}
        while len(recv_queue.keys()) < len(recv_queue_idxs):
            try:
                while len(recv_msg) < 4:
                    recv_msg += client_socket.recv(4)
                msg_size_bytes = recv_msg[:4]
                recv_msg = recv_msg[4:]
                total_recv_msg_size = struct.unpack('i', msg_size_bytes)[0]
                
                # Recv end signal - this might not happen
                if total_recv_msg_size == 0:
                    # send the end signal : 0
                    send_msg = struct.pack('i', 0)
                    client_socket.sendall(send_msg)
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
        # Back inference
        if len(back_models) == 0:
            out = recv_queue[recv_queue_idxs[0]]
        else:
            for in_idx in pass_queue_idxs:
                back_models[0].set_input("input_{}".format(in_idx), pass_queue[in_idx])
            for in_idx in recv_queue_idxs:
                back_models[0].set_input("input_{}".format(in_idx), recv_queue[in_idx])
                # print(recv_queue[in_idx])

            back_models[0].run()
            out = back_models[0].get_output(0).numpy()

        img_in_rgb = frame
        th = cv2.resize(cv2.threshold(np.squeeze(out.transpose([0,2,3,1])), 0.5, 1, cv2.THRESH_BINARY)[-1], (img_size,img_size))
        img_in_rgb[th == 1] = [0, 0, 255]

        if args.visualize:
            cv2.imshow("received - client", img_in_rgb)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Option : visualize
    print(time.time() - stime)

    print("------------------------")
