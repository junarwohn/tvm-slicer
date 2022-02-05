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
            return cv2.resize(img[490:1800, 900:2850], (im_sz,im_sz)) / 255
        return preprocess
    elif model == 'resnet152':
        def preprocess(img):
            return cv2.resize(img, (im_sz, im_sz))
        return preprocess

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

model_path = "../src/model/{}_{}_front_{}_{}_{}.so".format(args.model, args.target, args.img_size, args.opt_level, args.partition_point)
front_lib = tvm.runtime.load_module(model_path)
front_model = graph_executor.GraphModule(front_lib['default'](dev))

model_info_path = "../src/graph/{}_{}_front_{}_{}_{}.json".format(args.model, args.target, args.img_size, args.opt_level, args.partition_point)
with open(model_info_path, "r") as json_file:
    model_info = json.load(json_file)

input_info = model_info["extra"]["inputs"]
shape_info = model_info["attrs"]["shape"][1][:len(input_info)]
output_info = model_info["extra"]["outputs"]

# # Initialize connect

HOST_IP = args.ip
PORT = 9998       
#socket_size = 16 * 1024 * 1024
socket_size = args.socket_size

client_socket  = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
client_socket.connect((HOST_IP, PORT))

# initialize final output size
total_recv_bytes = struct.unpack('i', client_socket.recv(4))[0]
recv_msg = client_socket.recv(total_recv_bytes)
while len(recv_msg) < total_recv_bytes:
    recv_msg += client_socket.recv(total_recv_bytes)

final_output_shape = np.frombuffer(recv_msg, np.int).reshape((4,))
##
print(final_output_shape)


# Video Load
img_size = 512 
cap = cv2.VideoCapture("../src/data/j_scan.mp4")
# client_socket.settimeout(1)
total_time = 0
total_time_start = time.time()
inference_time = 0
network_time = 0

# timer INIT
timer_inference = 0
timer_total = 0
timer_exclude_network = 0

timer_toal_start = time.time()

while (cap.isOpened()):
    timer_exclude_network_start = time.time()
    ret, frame = cap.read()
    try:
        frame = preprocess(frame)
    except:
        total_msg = struct.pack('i', 0)
        client_socket.sendall(total_msg)
        client_socket.close()
        break
    
    input_data = np.expand_dims(frame, 0).transpose([0, 3, 1, 2])
    timer_inference_start = time.time()
    front_model.set_input("input_0", input_data)
    front_model.run()
    outs = []
    for i, out_idx in enumerate(output_info):
        out = front_model.get_output(i).asnumpy().astype(np.float32)
        outs.append(out)

    timer_inference += time.time() - timer_inference_start
    
    msg_body = b''
    # Send msg
    for out in outs:
        out_byte = out.tobytes()
        msg_body += out_byte
    
    timer_exclude_network += time.time() - timer_exclude_network_start
    total_send_msg_size = len(msg_body)
    send_msg = struct.pack('i', total_send_msg_size) + msg_body
    # Send object
    client_socket.sendall(send_msg)

    # Receive object
    total_recv_msg_size = struct.unpack('i', client_socket.recv(4))[0]
    if total_recv_msg_size == 0:
        break 
    
    recv_msg = client_socket.recv(total_recv_msg_size)
    while len(recv_msg) < total_recv_msg_size:
        recv_msg += client_socket.recv(total_recv_msg_size)

    recv_outs = []
    b,c,h,w = final_output_shape
    ## TODO : get output and parse 
    out = np.frombuffer(recv_msg[:4*b*c*h*w], np.float32).reshape(tuple(final_output_shape))

    timer_exclude_network_start = time.time()
    img_in_rgb = frame
    th = cv2.resize(cv2.threshold(np.squeeze(out.transpose([0,2,3,1])), 0.5, 1, cv2.THRESH_BINARY)[-1], (img_size,img_size))
    img_in_rgb[th == 1] = [0, 0, 255]
    cv2.imshow("received - client", img_in_rgb)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    timer_exclude_network += time.time() - timer_exclude_network_start 

timer_total = time.time() - timer_toal_start
timer_network = timer_total - timer_exclude_network

print("total time :", timer_total)
print("inference time :", timer_inference)
print("exclude network time :", timer_exclude_network)
print("network time :", timer_network)

print("data receive size :", total_recv_msg_size)
print("data send size :", total_send_msg_size)

cap.release()
cv2.destroyAllWindows()
