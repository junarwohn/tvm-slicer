import socket
import pickle
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
parser.add_argument('--device', type=str, default='cuda', help='type of devices [llvm, cuda]')
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

#model_path = "../src/model/unet_cuda_512_3.so"
# model_path = "../src/model/unet_llvm_512_3.so"
#model_path = "../src/model/unet_llvm_512_3_q.so"
#model_path = "../../tests/very_simple_model/unet_3_q.so"
model_path = "../src/model/unet_cuda_full_512_3_42.so"
lib = tvm.runtime.load_module(model_path)
model = graph_executor.GraphModule(lib['default'](dev))

# Video Load

img_size = 512 
cap = cv2.VideoCapture("../../src/data/j_scan.mp4")
# client_socket.settimeout(1)
stime = time.time()

# timer INIT
timer_inference = 0
timer_total = 0
timer_exclude_network = 0
total_recv_msg_size = 0
total_send_msg_size = 0

timer_toal_start = time.time()

while (cap.isOpened()):
    time_read_start = get_time(args.ntp_enable)
    ret, frame = cap.read()
    try:
        frame = preprocess(frame)    
    except:
        break
    input_data = np.expand_dims(frame, 0).transpose([0, 3, 1, 2])

    timer_inference_start = time.time()

    model.set_input("input_1", input_data)
    model.run()
    for i in range(2):
        outd = model.get_output(i)
        out = outd.numpy()
        # if i < 4:
        # print(i, out.flatten()[256:256 + 100])
    #outd = model.get_output(4)
    outd = model.get_output(4)
    out = outd.numpy().astype(np.float32)
    
    timer_inference += time.time() - timer_inference_start

    img_in_rgb = frame
    th = cv2.resize(cv2.threshold(np.squeeze(out.transpose([0,2,3,1])), 0.5, 1, cv2.THRESH_BINARY)[-1], (img_size,img_size))
    img_in_rgb[th == 1] = [0, 0, 255]
    if args.visualize:
        cv2.imshow("received - client", img_in_rgb)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

timer_total = time.time() - timer_toal_start
timer_network = timer_total - timer_exclude_network


print("total time :", timer_total)
print("inference time :", timer_inference)
# print("exclude network time :", timer_exclude_network)
# print("network time :", timer_network)

print("data receive size :", total_recv_msg_size)
print("data send size :", total_send_msg_size)

cap.release()
