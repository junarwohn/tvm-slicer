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

def read_and_inference(send_queue):
    # Load models
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

    timer_model = 0

    # Load video file
    cap = cv2.VideoCapture("../../../tvm-slicer/src/data/j_scan.mp4")
    
    in_data = {0 : 0}

    # timer_set_input = [0 for i in range(len(model_input_indexs))]
    # timer_get_output = [0 for i in range()]

    # Start loop
    # TODO : get_data fuction to make modulize getting frames
    # TODO : This version sends all intermediate output, should be changed afterwards
    fpss = []
    recv_msg = b''
    while (cap.isOpened()):
        s_start = time.time()
        ret, frame = cap.read()                      
        try:
            frame = preprocess(frame)
        except:
            send_queue.put({-1 : -1})
            break
        # TIMER MODEL - start
        time_start = time.time()
        input_data = np.expand_dims(frame, 0).transpose([0, 3, 1, 2])

        in_data[0] = input_data

        pre_outputs = []
        if len(models) == 0:
            pre_outputs = [0]
        for in_indexs, out_indexs, model in zip(model_input_indexs, model_output_indexs, models):
            # set input
            for input_index in in_indexs:
                model.set_input("input_{}".format(input_index), in_data[input_index])
            # run model
            model.run()

            if len(pre_outputs) != 0:
                # sync_send_img({k : in_data[k] for k in pre_outputs})
                send_queue.put({k : in_data[k] for k in pre_outputs})

            # get output
            for i, output_index in enumerate(out_indexs):
                in_data[output_index] = model.get_output(i).numpy()
            pre_outputs = out_indexs

        # sync_send_img({k : in_data[k] for k in pre_outputs})
        send_queue.put({k : in_data[k] for k in pre_outputs})

        # Timer stop
        timer_model += time.time() - time_start
        
        # Put data to send msg process
        # send_queue.put(outs)
        # frame_queue.put(frame)
        # Receive size of message (int - 4 byte)
        while len(recv_msg) < 4:
            recv_msg += client_socket.recv(4)

        msg_size_bytes = recv_msg[:4]
        recv_msg = recv_msg[4:]
        total_recv_msg_size = struct.unpack('i', msg_size_bytes)[0]

        # Exit condition
        if total_recv_msg_size == 0:
            break 

        # Receive data object
        while len(recv_msg) < total_recv_msg_size:
            recv_msg += client_socket.recv(total_recv_msg_size)

        ## TODO : get output and parse 
        msg_data_bytes = recv_msg[:total_recv_msg_size]
        data = pickle.loads(msg_data_bytes)
        recv_msg = recv_msg[total_recv_msg_size:]

        outs = []
        for key in data.keys():
            outs.append(data[key])
        
        img_in_rgb = frame
        th = cv2.resize(cv2.threshold(np.squeeze(outs[0].transpose([0,2,3,1])), 0.5, 1, cv2.THRESH_BINARY)[-1], (img_size,img_size))
        img_in_rgb[th == 1] = [0, 0, 255]
        if args.visualize:
            cv2.imshow("received - client", img_in_rgb)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # print('recv_img End')

        fpss.append(1/(time.time() - s_start))
    cv2.destroyAllWindows()
    print("Mean FPS :", np.mean(fpss))
    client_socket.close()


def sync_send_img(data):
    # Packing data

    msg_body = pickle.dumps(data)
    total_send_msg_size = len(msg_body)
    send_msg = struct.pack('i', total_send_msg_size) + msg_body

    # Send object
    client_socket.sendall(send_msg)
    

async def async_send_img(data):
    # Packing data

    msg_body = pickle.dumps(data)
    total_send_msg_size = len(msg_body)
    send_msg = struct.pack('i', total_send_msg_size) + msg_body

    # Send object
    client_socket.sendall(send_msg)
        

def send_img(send_queue):
    while True:
        if not send_queue.empty():
            # Get data
            data = send_queue.get()

            # exit codition : {-1 : -1}
            if -1 in data.keys():
                while send_queue.qsize() != 0:
                    # print("Exit Condition")
                    # Clean send_queue
                    send_queue.get()
                send_msg = struct.pack('i', 0)
                client_socket.sendall(send_msg)
                break
            
            # Packing data
            msg_body = pickle.dumps(data)
            total_send_msg_size = len(msg_body)
            send_msg = struct.pack('i', total_send_msg_size) + msg_body

            # Send object
            client_socket.sendall(send_msg)
        
    # Exit
    # client_socket.close()
    # print('send_img End')


def recv_img(frame_queue):
    recv_msg = b''
    while True:
        # Receive size of message (int - 4 byte)
        while len(recv_msg) < 4:
            recv_msg += client_socket.recv(4)

        msg_size_bytes = recv_msg[:4]
        recv_msg = recv_msg[4:]
        total_recv_msg_size = struct.unpack('i', msg_size_bytes)[0]

        # Exit condition
        if total_recv_msg_size == 0:
            break 

        # Receive data object
        while len(recv_msg) < total_recv_msg_size:
            recv_msg += client_socket.recv(total_recv_msg_size)

        ## TODO : get output and parse 
        msg_data_bytes = recv_msg[:total_recv_msg_size]
        data = pickle.loads(msg_data_bytes)
        recv_msg = recv_msg[total_recv_msg_size:]

        outs = []
        for key in data.keys():
            outs.append(data[key])
        
        img_in_rgb = frame_queue.get()
        th = cv2.resize(cv2.threshold(np.squeeze(outs[0].transpose([0,2,3,1])), 0.5, 1, cv2.THRESH_BINARY)[-1], (img_size,img_size))
        img_in_rgb[th == 1] = [0, 0, 255]

        if args.visualize:
            cv2.imshow("received - client", img_in_rgb)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    # print('recv_img End')
    cv2.destroyAllWindows()

if __name__ == '__main__':
    print("------------------------")
    print(args.model, ", ", args.target, ", ", args.img_size, ", ", args.opt_level, ", ", 'partition points :', args.partition_points, sep='')
    send_queue = Queue()
    p1 = Process(target=read_and_inference, args=(send_queue,))
    p2 = Process(target=send_img, args=(send_queue,))
    p1.start() 
    p2.start()
    stime = time.time()
    p1.join(); 
    p2.join(); 
    print("Total Time :", time.time() - stime)
    print("------------------------")
