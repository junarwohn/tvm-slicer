from email import message_from_binary_file
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

ntp_time_server = 'time.windows.com'               # NTP Server Domain Or IP 
ntp_time_server = 'time.google.com'               # NTP Server Domain Or IP 
g_ntp_client = ntplib.NTPClient() 
#response = c.request(ntp_time_server, version=3) 

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


# model_info_path = "../src/graph/{}_{}_front_{}_{}_{}.json".format(args.model, args.target, args.img_size, args.opt_level, args.partition_point)
# with open(model_info_path, "r") as json_file:
#     model_info = json.load(json_file)

# input_info = model_info["extra"]["inputs"]
# shape_info = model_info["attrs"]["shape"][1][:len(input_info)]
# output_info = model_info["extra"]["outputs"]
# dltype_info = [model_info["attrs"]["dltype"][1][output_idx] for output_idx in output_info]
# print(input_info, shape_info, output_info, dltype_info)
# Video Load
img_size = 512 

HOST_IP = args.ip
PORT = 9998       
#socket_size = 16 * 1024 * 1024
socket_size = args.socket_size

client_socket  = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
client_socket.connect((HOST_IP, PORT))

org=(50,100)
font=cv2.FONT_HERSHEY_SIMPLEX


def generate_img(q):

    model_path = "../src/model/{}_{}_front_0.so".format(args.model, args.target)
    lib = tvm.runtime.load_module(model_path)
    model0 = graph_executor.GraphModule(lib['default'](dev))

    model_path = "../src/model/{}_{}_front_1.so".format(args.model, args.target)
    lib = tvm.runtime.load_module(model_path)
    model1 = graph_executor.GraphModule(lib['default'](dev))

    model_path = "../src/model/{}_{}_front_2.so".format(args.model, args.target)
    lib = tvm.runtime.load_module(model_path)
    model2 = graph_executor.GraphModule(lib['default'](dev))

    model_path = "../src/model/{}_{}_front_3.so".format(args.model, args.target)
    lib = tvm.runtime.load_module(model_path)
    model3 = graph_executor.GraphModule(lib['default'](dev))

    # model_path = "../src/model/{}_{}_front_{}_{}_{}.so".format(args.model, args.target, args.img_size, args.opt_level, args.partition_point)
    # # model_path = "../src/model/{}_{}_full_{}_{}_{}.so".format(args.model, args.target, args.img_size, args.opt_level, args.partition_point)
    # front_lib = tvm.runtime.load_module(model_path)
    # front_model = graph_executor.GraphModule(front_lib['default'](dev))
    timer_model = 0
    cap = cv2.VideoCapture("../../../tvm-slicer/src/data/j_scan.mp4")
    while (cap.isOpened()):
        ret, frame = cap.read()
        try:
            frame = preprocess(frame)
        except:
            total_msg = struct.pack('i', 0)
            client_socket.sendall(total_msg)
            client_socket.close()
            break
        time_start = time.time()
        # print("imread")
        input_data = np.expand_dims(frame, 0).transpose([0, 3, 1, 2])
        
        # ----------------------------
        # # model 0 Inference Part
        # ----------------------------
        model0.set_input("input_0", input_data)
        model0.run()
        outd = model0.get_output(0)
        out0 = outd.numpy().astype(np.int8)
        # ----------------------------

        # ----------------------------
        # # model 1 Inference Part
        # ----------------------------
        model1.set_input("input_9", out0)
        model1.run()
        # ============================
        # # # Send model 0's output 
        msg_body = pickle.dumps([9, out0])
        total_send_msg_size = len(msg_body)
        send_msg = struct.pack('i', total_send_msg_size) + msg_body
        client_socket.sendall(send_msg)
        # ============================
        outd = model1.get_output(0)
        out1 = outd.numpy().astype(np.int8)
        # ----------------------------

        # ----------------------------
        # # model 2 Inference Part
        # ----------------------------
        model2.set_input("input_20", out1)
        model2.run()
        # ============================
        # # # Send model 1's output 
        msg_body = pickle.dumps([20, out1])
        total_send_msg_size = len(msg_body)
        send_msg = struct.pack('i', total_send_msg_size) + msg_body
        client_socket.sendall(send_msg)
        # ============================
        outd = model2.get_output(0)
        out2 = outd.numpy().astype(np.int8)
        # ----------------------------

        # ----------------------------
        # model 3 Inference Part
        # ----------------------------
        model3.set_input("input_31", out2)
        model3.run()
        # ============================
        # # # Send model 2's output 
        msg_body = pickle.dumps([31, out2])
        total_send_msg_size = len(msg_body)
        send_msg = struct.pack('i', total_send_msg_size) + msg_body
        client_socket.sendall(send_msg)
        # ============================
        outd = model3.get_output(0)
        out3 = outd.numpy().astype(np.int8)
        # ----------------------------
        

        # ============================
        # # # Send model 3's output 
        msg_body = pickle.dumps([42, out3])
        total_send_msg_size = len(msg_body)
        send_msg = struct.pack('i', total_send_msg_size) + msg_body
        client_socket.sendall(send_msg)
        # ============================
        q.put(frame)

        # print("model run")
        timer_model += time.time() - time_start

        # print("send")
    print(timer_model)
    client_socket.close()

    
def recv_img(q):
    recv_msg = b''
    while True:
        # print("recv")
        while len(recv_msg) < 4:
            # print("recv")
            recv_msg += client_socket.recv(4)
        total_recv_msg_size = struct.unpack('i', recv_msg[:4])[0]
        recv_msg = recv_msg[4:]
        if total_recv_msg_size == 0:
            break 
        # print("total_recv_msg_size", total_recv_msg_size)
        # recv_msg += client_socket.recv(total_recv_msg_size)
        while len(recv_msg) < total_recv_msg_size:
            # print(len(recv_msg))
            recv_msg += client_socket.recv(total_recv_msg_size)
        # img = np.frombuffer(recv_msg[:4*512*512*3], np.float32).reshape((512,512,3))

        ## TODO : get output and parse 
        out = pickle.loads(recv_msg[:total_recv_msg_size])
        img_in_rgb = q.get()
        # print(out.flatten()[:10])
        th = cv2.resize(cv2.threshold(np.squeeze(out.transpose([0,2,3,1])), 0.5, 1, cv2.THRESH_BINARY)[-1], (img_size,img_size))
        img_in_rgb[th == 1] = [0, 0, 255]

        if args.visualize:
            # print("recv")
            cv2.imshow("received - client", img_in_rgb)
            # # cv2.imshow("received - client", 255 * th)
            # # # print(th)
            # ## cv2.waitKey(1)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        recv_msg = recv_msg[total_recv_msg_size:]

    cv2.destroyAllWindows()

if __name__ == '__main__':
    q = Queue()
    p1 = Process(target=generate_img, args=(q,))
    p2 = Process(target=recv_img, args=(q,))
    p1.start(); 
    p2.start(); 
    stime = time.time()
    p1.join(); p2.join()
    print(time.time() - stime)
