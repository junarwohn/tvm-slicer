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
parser.add_argument('--ip', type=str, default='192.168.0.184', help='input ip of host')
parser.add_argument('--device', type=str, default='cuda', help='type of devices [llvm, cuda]')
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

img_size = args.img_size

model_path = "../src/model/{}_{}_{}_{}.so".format(args.model, args.target, img_size, args.opt_level)
lib = tvm.runtime.load_module(model_path)
model = graph_executor.GraphModule(lib['default'](dev))

# Video Load

img_size = 512 
cap = cv2.VideoCapture("../src/data/j_scan.mp4")
# client_socket.settimeout(1)
stime = time.time()

total_time = 0
total_time_start = time.time()
inference_time = 0
network_time = 0


# timer INIT
timer_READ = 0
timer_SET_INPUT = 0
timer_RUN_MODEL = 0
timer_GET_OUTPUT = 0
timer_ASNUMPY = 0
timer_VISUALIZE = 0


# TVM inference sync
#inference_sync = False
inference_sync = True

outd = tvm.nd.empty((1,1,512,512), dtype='float32', device=dev)

## WARM UP
dummy_input = np.random.normal(0,1,(1,3,512,512))
for i in range(10):
    model.set_input("input_1", dummy_input)
    dev.sync()
    model.run()
    dev.sync()
    outd = model.get_output(0)
    dev.sync()
##

while (cap.isOpened()):
    ### timer : READ start
    timer_start = time.time()
    ret, frame = cap.read()
    try:
        frame = cv2.resize(frame[490:1800, 900:2850], (img_size, img_size)) / 255
    except:
        break
    input_data = np.expand_dims(frame, 0).transpose([0, 3, 1, 2])
    ### timer : READ end 
    timer_READ += time.time() - timer_start 



    ### timer : SET_INPUT start
    timer_start = time.time()
    # Execute front
    model.set_input("input_1", input_data)
    dev.sync()
    ### timer : SET_INPUT end 
    timer_SET_INPUT += time.time() - timer_start 


    ### timer : RUN_MODEL start
    timer_start = time.time()
    model.run()
    dev.sync()
    ### timer : RUN_MODEL end 
    timer_RUN_MODEL += time.time() - timer_start


    ### timer : GET_OUTPUT start
    timer_start= time.time()
    outd = model.get_output(0)
    dev.sync()
    #outd = model.get_output(0, outd)

    ### timer : GET_OUTPUT end
    timer_GET_OUTPUT += time.time() - timer_start


    ### timer : ASNUMPY start
    timer_start = time.time()
    out = outd.asnumpy()
    timer_ASNUMPY += time.time() - timer_start
    dev.sync()


    # # TIME_CHECK : VISUALIZE 
    # timer_start = time.time() 
    # img_in_rgb = frame
    # th = cv2.resize(cv2.threshold(np.squeeze(out.transpose([0,2,3,1])), 0.5, 1, cv2.THRESH_BINARY)[-1], (img_size,img_size))
    # img_in_rgb[th == 1] = [0, 0, 255]
    # cv2.imshow("received - client", img_in_rgb)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break
    # # ### TIME_CHECK : VISUALIZE END
    # timer_VISUALIZE += time.time() - timer_start


total_time_checker = 0

#timer_SET_INPUT 
#timer_RUN_MODEL
#timer_GET_OUTPUT
#timer_ASNUMPY
#timer_VISUALIZE
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(total_frames)

print('INFERNECE SYNC', ':', inference_sync)
print("Total time :", time.time() - stime)
print('SET_INPUT per frame (ms)', ':', timer_SET_INPUT / total_frames * 1000)
print('RUN_MODEL per frame (ms)', ':', timer_RUN_MODEL / total_frames * 1000)
print('GET_OUTPUT per frame (ms)', ':', timer_GET_OUTPUT / total_frames * 1000)
print('ASNUMPY per frame (ms)', ':', timer_ASNUMPY / total_frames * 1000)
print('VISUALIZE', ':', timer_VISUALIZE)

cap.release()
